import openai
from lcb_runner.runner.base_runner import BaseRunner
import asyncio
import json
from tqdm.asyncio import tqdm


import time

import openai
from openai.types.chat import ChatCompletion


async def make_request(
    client: openai.AsyncClient,
    prompt: str,
    model: str,
    **kwargs
) -> ChatCompletion:
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )


async def make_auto_request(*args, **kwargs) -> ChatCompletion:
    completion = None
    while completion is None:
        try:
            completion = await make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            await asyncio.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            await asyncio.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            await asyncio.sleep(1)
    return completion


class VLLMServerRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.model_name = model.model_name if args.local_model_path is None else args.local_model_path
        self.enable_thinking = args.enable_thinking
        self.client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)
        self.client_kwargs = {
            "model": self.model_name,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "n": self.args.n,
        }
        if not self.enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            self.client_kwargs["extra_body"] = extra_body

    async def _run_single(self, prompt: str) -> list[str]:
        completion = await make_auto_request(
            self.client,
            prompt,
            **self.client_kwargs
        )
        return [c.message.content for c in completion.choices]

    def run_batch(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
        async def run_all():
            tasks = []
            for prompt in prompts:
                if isinstance(prompt, list):
                    prompt_str = json.dumps(prompt)
                else:
                    prompt_str = prompt
                tasks.append(self._run_single(prompt_str))
            return await tqdm.gather(*tasks)

        outputs = asyncio.run(run_all())

        if self.args.use_cache:
            for prompt, output in zip(prompts, outputs):
                if isinstance(prompt, list):
                    prompt_cache = json.dumps(prompt)
                elif isinstance(prompt, tuple):
                    prompt_cache = prompt[0] + json.dumps(prompt[1])
                else:
                    prompt_cache = prompt
                self.cache[prompt_cache] = output  ## save the output to cache

        return outputs


