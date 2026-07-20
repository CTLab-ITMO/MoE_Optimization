import os
import json
import logging

import torch
import argparse

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    # args = get_args()

    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    # for i in range(len(combined_results)):
    #     for j in range(len(combined_results[i][1])):
    #         if "def solve()" in combined_results[i][1][j]:
    #             from lcb_runner.utils.extraction_utils import extract_code, LMStyle

    #             combined_results[i][1][j] = extract_code(
    #                 combined_results[i][0][j], LMStyle.Gemini
    #             )
    #             if "\nsolve()" not in combined_results[i][1][j]:
    #                 combined_results[i][1][j] += "\n\nsolve()"

    #                 # combined_results[i][1][j] += "\n\nsolve()"
    #                 print(combined_results[i][1][j])

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)


def get_args_dict(
    model="gpt-3.5-turbo-0301",
    local_model_path=None,
    trust_remote_code=False,
    scenario=Scenario.codegeneration,
    not_fast=False,
    release_version="release_latest",
    cot_code_execution=False,
    n=10,
    codegen_n=10,
    temperature=0.2,
    top_p=0.95,
    max_tokens=16384,
    multiprocess=0,
    stop="###",
    continue_existing=False,
    continue_existing_with_eval=False,
    use_cache=False,
    cache_batch_size=100,
    debug=False,
    evaluate=False,
    num_process_evaluate=12,
    timeout=60,
    openai_timeout=90,
    tensor_parallel_size=-1,
    enable_prefix_caching=False,
    custom_output_file=None,
    custom_output_save_name=None,
    output_path="output",
    dtype="bfloat16",
    start_date=None,
    end_date=None,
    enable_thinking=False,
    base_url="http://0.0.0.0:8000/v1/completions",
):
    if isinstance(scenario, str):
        scenario = Scenario[scenario]
    args = argparse.Namespace(
        model=model,
        local_model_path=local_model_path,
        trust_remote_code=trust_remote_code,
        scenario=scenario,
        not_fast=not_fast,
        release_version=release_version,
        cot_code_execution=cot_code_execution,
        n=n,
        codegen_n=codegen_n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        multiprocess=multiprocess,
        stop=stop,
        continue_existing=continue_existing,
        continue_existing_with_eval=continue_existing_with_eval,
        use_cache=use_cache,
        cache_batch_size=cache_batch_size,
        debug=debug,
        evaluate=evaluate,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
        openai_timeout=openai_timeout,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=enable_prefix_caching,
        custom_output_file=custom_output_file,
        custom_output_save_name=custom_output_save_name,
        output_path=output_path,
        dtype=dtype,
        start_date=start_date,
        end_date=end_date,
        enable_thinking=enable_thinking,
        base_url=base_url,
    )

    args.stop = args.stop.split(",")

    if args.tensor_parallel_size == -1:
        args.tensor_parallel_size = torch.cuda.device_count()

    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()

    return args



if __name__ == "__main__":
    args = get_args()
    main(args)
