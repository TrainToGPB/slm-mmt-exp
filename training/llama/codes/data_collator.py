"""
Get from trl/DataCollatorForCompletionOnlyLM
"""
# built-in
import warnings
from typing import Any, Dict, List, Union

# third-party
import numpy as np
from transformers import DataCollatorForLanguageModeling


class Llama2DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        instruction_template (`Optional[str]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Union[str, List[int]] = None,
        lang_table: Dict[str, str] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
            if self.response_token_ids[0] == 29871:
                self.response_token_ids = self.response_token_ids[1:]
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.lang_table = lang_table

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None
                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                    
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                for lang in self.lang_table.values():
                    src_lang_sign = f'### {lang}-source'
                    src_lang_token_ids = self.tokenizer.encode(src_lang_sign, add_special_tokens=False)
                    
                    src_lang_idx = None
                    for idx in range(len(batch["labels"][i]) - len(src_lang_token_ids) + 1):
                        if np.array_equal(batch["labels"][i][idx:idx+len(src_lang_token_ids)], src_lang_token_ids):
                            src_lang_idx = idx
                            break

                    if src_lang_idx is not None:
                        src_lang = lang
                        continue

                    tgt_lang_sign = f'### {lang}-translated'
                    tgt_lang_token_ids = self.tokenizer.encode(tgt_lang_sign, add_special_tokens=False)
                    
                    tgt_lang_idx = None
                    for idx in range(len(batch["labels"][i]) - len(tgt_lang_token_ids) + 1):
                        if np.array_equal(batch["labels"][i][idx:idx+len(tgt_lang_token_ids)], tgt_lang_token_ids):
                            tgt_lang_idx = idx
                            break

                    if tgt_lang_idx is not None:
                        tgt_lang = lang
                        continue

                src_token_ids = self.tokenizer.encode(src_lang, add_special_tokens=False)
                tgt_lang_token_ids = self.tokenizer.encode(tgt_lang, add_special_tokens=False)

                src_special_token_id = self.tokenizer.convert_tokens_to_ids(["<|src|>"])[0]
                tgt_lang_special_token_id = self.tokenizer.convert_tokens_to_ids(["<|tgt|>"])[0]

                response_token_ids_with_tgt = self.response_token_ids
                added_len = 0
                for idx in np.where(np.array(response_token_ids_with_tgt) == tgt_lang_special_token_id)[0]:
                    response_token_ids_with_tgt = response_token_ids_with_tgt[:idx+added_len] + tgt_lang_token_ids + response_token_ids_with_tgt[idx+1+added_len:]
                    added_len += len(tgt_lang_token_ids) - 1
                self.response_template = self.tokenizer.decode(response_token_ids_with_tgt, skip_special_tokens=True)

                instruction_token_ids = self.instruction_token_ids
                added_len = 0
                for idx in np.where(np.array(instruction_token_ids) == src_special_token_id)[0]:
                    instruction_token_ids = instruction_token_ids[:idx+added_len] + src_token_ids + instruction_token_ids[idx+1+added_len:]
                    added_len += len(src_token_ids) - 1
                added_len = 0
                for idx in np.where(np.array(instruction_token_ids) == tgt_lang_special_token_id)[0]:
                    instruction_token_ids = instruction_token_ids[:idx+added_len] + tgt_lang_token_ids + instruction_token_ids[idx+1+added_len:]
                    added_len += len(tgt_lang_token_ids) - 1
                self.instruction_template = self.tokenizer.decode(instruction_token_ids, skip_special_tokens=True)

                response_token_ids_idxs = []
                human_token_ids_idxs = []
                for assistant_idx in np.where(batch["labels"][i] == response_token_ids_with_tgt[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        response_token_ids_with_tgt
                        == batch["labels"][i][assistant_idx : assistant_idx + len(response_token_ids_with_tgt)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(response_token_ids_with_tgt))
                
                if len(response_token_ids_idxs) == 0:
                    start_idx = np.where(batch["labels"][i] == response_token_ids_with_tgt[0])[0][0]
                    warnings.warn(
                        f"\nSOURCE: {src_lang} / TARGET: {tgt_lang}\n"
                        f"RESPONSE: {self.response_template}\n"
                        f"RESPONSE IDS: {self.response_token_ids}\n"
                        f"LABEL: {self.tokenizer.decode(batch['input_ids'][i][start_idx:start_idx+len(self.response_token_ids)+1], skip_special_tokens=True)}\n"
                        f"LABEL IDS: {batch['labels'][i][start_idx:start_idx+len(self.response_token_ids)+1]}\n"
                        f"Could not find response key\n`{self.response_template}`\nin the "
                        f'following instance:\n{self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)}\n'
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.\n\n"
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    start_idx = 1
                    warnings.warn(
                        f"SOURCE: {src_lang} / TARGET: {tgt_lang}\n"
                        f"INSTRUCTION: {self.instruction_template}\n"
                        f"INSTRUCTION IDS: {human_token_ids}\n"
                        f"LABEL: {self.tokenizer.decode(batch['input_ids'][i][start_idx:start_idx+len(human_token_ids)+1], skip_special_tokens=True)}\n"
                        f"LABEL IDS: {batch['labels'][i][start_idx:start_idx+len(human_token_ids)+1]}\n"
                        f"Could not find instruction key\n`{self.instruction_template}`\nin the "
                        f'following instance:\n{self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)}\n'
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.\n\n"
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch


class Llama3DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
        instruction_template (`Optional[str]`): the template form that indicates the start of the human instruction, typically something like 
            '### Human:\n'. Useful for assistant-style conversation datasets (default: None)
        lang_table (`Dict[str, str]`): the table that maps the language names to the language codes (default: None)
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
            for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`): The index to use to ignore the initial tokens with (default: -100)
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Union[str, List[int]] = None,
        lang_table: Dict[str, str] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.lang_table = lang_table

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None
                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                    
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                batch_to_decode = np.delete(batch["labels"][i], np.where(batch["labels"][i] == self.ignore_index))
                batch_decoded = self.tokenizer.decode(batch_to_decode, skip_special_tokens=True)
                tgt_lang = None
                for lang in self.lang_table.values():
                    tgt_lang_sign = f'<target><{lang}>'
                    if tgt_lang_sign in batch_decoded:
                        tgt_lang = lang
                        break
                
                if tgt_lang is None:
                    batch["labels"][i, :] = self.ignore_index
                    warnings.warn(f"Could not find target language sign in the label")

                response_token_ids_with_tgt = self.response_token_ids
                response_with_tgt = self.response_template.replace('<|tgt|>', f'{tgt_lang}')
                response_token_ids_with_tgt = self.tokenizer.encode(response_with_tgt, add_special_tokens=False)

                instruction_token_ids = self.instruction_token_ids
                self.instruction_template = self.tokenizer.decode(instruction_token_ids, skip_special_tokens=True)

                response_token_ids_idxs = []
                human_token_ids_idxs = []
                assistant_idx = None
                for idx in range(len(batch["labels"][i]) - len(response_token_ids_with_tgt) + 1):
                    batch_for_check = batch['labels'][i][idx:idx+len(response_token_ids_with_tgt)]
                    if self.ignore_index in batch_for_check:
                        continue
                    if np.array_equal(batch_for_check, response_token_ids_with_tgt):
                        assistant_idx = idx
                        break
                if (assistant_idx is not None):
                    if (
                        response_token_ids_with_tgt
                        == batch["labels"][i][assistant_idx : assistant_idx + len(response_token_ids_with_tgt)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(response_token_ids_with_tgt))
                
                if len(response_token_ids_idxs) == 0:
                    batch_for_print = np.delete(batch["labels"][i], np.where(batch["labels"][i] == self.ignore_index))
                    warnings.warn(
                        f"\n[NO RESPONSE SUFFIX MATCHED]\n"
                        f"Could not find response key `{self.tokenizer.decode(response_token_ids_with_tgt, skip_special_tokens=True)}` in the "
                        f'following instance:\n{self.tokenizer.decode(batch_for_print, skip_special_tokens=True)}\n'
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.\n\n"
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = instruction_token_ids
                for idx in range(len(batch["labels"][i]) - len(human_token_ids) + 1):
                    if np.array_equal(batch["labels"][i][idx:idx+len(human_token_ids)], human_token_ids):
                        human_token_ids_idxs.append(idx)
                        break

                if len(human_token_ids_idxs) == 0:
                    batch_for_print = np.delete(batch["labels"][i], np.where(batch["labels"][i] == self.ignore_index))
                    warnings.warn(
                        f"\n[NO INSTRUCTION SUFFIX MATCHED]\n"
                        f"Could not find instruction key\n`{self.tokenizer.decode(instruction_token_ids, skip_special_tokens=True)}`\nin the "
                        f'following instance:\n{self.tokenizer.decode(batch_for_print, skip_special_tokens=True)}\n'
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.\n\n"
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch
    