from MoE_Optimization.common.input_data_utils.mmlu_batch_generator import MMLUBatchGenerator


def convert_text_inputs_to_tokens(gen: MMLUBatchGenerator, tokenizer, device):
   converted_batches = []
   for batch in gen:
        inputs = tokenizer(batch, return_tensors="pt",
                padding=True,  # Pad to longest in batch
                truncation=True,
                max_length=512,  # Adjust based on your needs
                add_special_tokens=True
        )
        converted_batches.append({k: v.to(device) for k, v in inputs.items()})
   return converted_batches