import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.modules.transforms import Transform
from torchtune.datasets._sft import SFTTransform
from typing import Any, Callable, Optional, Union
from torchtune.data._messages import OpenAIToMessages, ShareGPTToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
import random

class NTPMHuBERT1000Dataset(Dataset):
    """
    将数据集前半部分作为语音、后半部分作为动作进行成对取样，并做随机对齐裁剪。
    通过传入的 message_transform（ShareGPT/OpenAI）将样本转换为 messages，
    再用 tokenizer（ModelTokenizer）完成 tokenization，保持与 SFTDataset 一致的输出字典键。
    """

    def __init__(
        self,
        *,
        tokenizer: ModelTokenizer,
        source: str,
        message_transform: Transform,
        split: str = "train",
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        audio_text_column: str = "text",
        motion_text_column: str = "text",
        repeat_factor: int = 200,
        cutoff_audio_length: int = 200,
        cutoff_motion_length: int = 30,
        audio_stride: int = 20,
        motion_stride: int = 3,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = tokenizer

        self._data = load_dataset(source, split=split, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

        self.audio_text_column = audio_text_column
        self.motion_text_column = motion_text_column
        self.repeat_factor = repeat_factor
        self.cutoff_audio_length = cutoff_audio_length
        self.cutoff_motion_length = cutoff_motion_length
        self.audio_stride = audio_stride
        self.motion_stride = motion_stride

        self.pair_num = len(self._data) // 2
        if self.pair_num == 0:
            raise ValueError("数据量不足，无法形成语音-动作配对。")

        # 与 SFTDataset 相同的预处理流水线：message_transform -> model_transform
        self._prepare_sample = SFTTransform(
            message_transform=self._message_transform,
            model_transform=self._model_transform,
        )

    def __len__(self) -> int:
        # 防止因为数据集太小了导致一次迭代不了几次
        return len(self._data) * self.repeat_factor

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index%(self.pair_num*2)]
        if sample['type']=='motion_token':
            token_len = self.cutoff_motion_length
            sample_column = self.motion_text_column
        elif sample['type']=='audio_token':
            token_len = self.cutoff_audio_length
            sample_column = self.audio_text_column

        sample_tokens = sample[sample_column]


        # 随机对齐裁剪（对齐到 stride）
        rand_range = max(1, len(sample_tokens) - token_len + 1)
        start_idx = np.random.randint(0, rand_range)
        end_idx = min(start_idx + token_len, len(sample_tokens))
        sample_cut = sample_tokens[start_idx:end_idx]
        tokenized_dict = {}
        tokenized_dict['tokens'] = self._model_transform.encode("".join(sample_cut))
        tokenized_dict["mask"] = [False] * len(tokenized_dict['tokens'])
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"][1:],
                -100,
                tokenized_dict["tokens"][1:],
            )
        )
        tokenized_dict["labels"].append(-100)

        # 复用 SFTTransform：生成 tokens/mask，并自动构造 labels（带位移与忽略）
        return tokenized_dict


def ntp_dataset_mhubert_1000(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_style: str = "openai",
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    audio_text_column: str = "text",
    motion_text_column: str = "text",
    repeat_factor: int = 200,
    cutoff_audio_length: int = 200,
    cutoff_motion_length: int = 30,
    audio_stride: int = 20,
    motion_stride: int = 3,
    **load_dataset_kwargs: dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    与现有 chat_dataset 风格一致的构建器：选择 ShareGPT/OpenAI 消息格式，
    并返回可选打包的 Dataset（PackedDataset）。
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": "conversations"},
            new_system_prompt=new_system_prompt,
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": "messages"},
            new_system_prompt=new_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = NTPMHuBERT1000Dataset(
        tokenizer=tokenizer,
        source=source,
        message_transform=message_transform,
        split=split,
        filter_fn=filter_fn,
        audio_text_column=audio_text_column,
        motion_text_column=motion_text_column,
        repeat_factor=repeat_factor,
        cutoff_audio_length=cutoff_audio_length,
        cutoff_motion_length=cutoff_motion_length,
        audio_stride=audio_stride,
        motion_stride=motion_stride,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds




class SFTMHuBERT1000Dataset(Dataset):
    """
    将数据集前半部分作为语音、后半部分作为动作进行成对取样，并做随机对齐裁剪。
    通过传入的 message_transform（ShareGPT/OpenAI）将样本转换为 messages，
    再用 tokenizer（ModelTokenizer）完成 tokenization，保持与 SFTDataset 一致的输出字典键。
    """

    def __init__(
        self,
        *,
        tokenizer: ModelTokenizer,
        source: str,
        message_transform: Transform,
        split: str = "train",
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        audio_text_column: str = "text",
        motion_text_column: str = "text",
        repeat_factor: int = 200,
        cutoff_audio_length: int = 200,
        cutoff_motion_length: int = 30,
        audio_stride: int = 20,
        motion_stride: int = 3,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = tokenizer

        self._data = load_dataset(source, split=split, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

        self.audio_text_column = audio_text_column
        self.motion_text_column = motion_text_column
        self.repeat_factor = repeat_factor
        self.cutoff_audio_length = cutoff_audio_length
        self.cutoff_motion_length = cutoff_motion_length
        self.audio_stride = audio_stride
        self.motion_stride = motion_stride

        self.pair_num = len(self._data) // 2
        if self.pair_num == 0:
            raise ValueError("数据量不足，无法形成语音-动作配对。")

        # 与 SFTDataset 相同的预处理流水线：message_transform -> model_transform
        self._prepare_sample = SFTTransform(
            message_transform=self._message_transform,
            model_transform=self._model_transform,
        )

    def __len__(self) -> int:
        # 防止因为数据集太小了导致一次迭代不了几次
        return len(self._data) * self.repeat_factor

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = index % self.pair_num
        sample_audio = self._data[idx]
        sample_motion = self._data[idx + self.pair_num]

        audio_tokens = sample_audio[self.audio_text_column]
        motion_tokens = sample_motion[self.motion_text_column]

        if not isinstance(audio_tokens, (list, tuple)) or not isinstance(
            motion_tokens, (list, tuple)
        ):
            raise TypeError(
                f"期望列 '{self.audio_text_column}' 与 '{self.motion_text_column}' 为 list/tuple，而不是其他类型。"
            )

        # 随机对齐裁剪（对齐到 stride）
        rand_range = max(1, len(audio_tokens) - self.cutoff_audio_length + 1)
        audio_start_idx = np.random.randint(0, rand_range)
        audio_start_idx = (audio_start_idx // self.audio_stride) * self.audio_stride
        audio_end_idx = min(audio_start_idx + self.cutoff_audio_length, len(audio_tokens))
        audio_cut = audio_tokens[audio_start_idx:audio_end_idx]

        motion_start_idx = (audio_start_idx // self.audio_stride) * self.motion_stride
        motion_end_idx = min(motion_start_idx + self.cutoff_motion_length, len(motion_tokens))
        motion_cut = motion_tokens[motion_start_idx:motion_end_idx]

        if isinstance(self._message_transform, OpenAIToMessages):
            sample = {
                "messages": [
                    {"role": "user", "content": "".join(audio_cut)},
                    {"role": "assistant", "content": "".join(motion_cut)},
                ]
            }

        # 复用 SFTTransform：生成 tokens/mask，并自动构造 labels（带位移与忽略）
        return self._prepare_sample(sample)


def sft_dataset_mhubert_1000(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_style: str = "openai",
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    audio_text_column: str = "text",
    motion_text_column: str = "text",
    repeat_factor: int = 200,
    cutoff_audio_length: int = 200,
    cutoff_motion_length: int = 30,
    audio_stride: int = 20,
    motion_stride: int = 3,
    **load_dataset_kwargs: dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    与现有 chat_dataset 风格一致的构建器：选择 ShareGPT/OpenAI 消息格式，
    并返回可选打包的 Dataset（PackedDataset）。
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": "conversations"},
            new_system_prompt=new_system_prompt,
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": "messages"},
            new_system_prompt=new_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = SFTMHuBERT1000Dataset(
        tokenizer=tokenizer,
        source=source,
        message_transform=message_transform,
        split=split,
        filter_fn=filter_fn,
        audio_text_column=audio_text_column,
        motion_text_column=motion_text_column,
        repeat_factor=repeat_factor,
        cutoff_audio_length=cutoff_audio_length,
        cutoff_motion_length=cutoff_motion_length,
        audio_stride=audio_stride,
        motion_stride=motion_stride,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds



class ICLMHuBERT1000Dataset(Dataset):
    """
    将数据集前半部分作为语音、后半部分作为动作进行成对取样，并做随机对齐裁剪。
    通过传入的 message_transform（ShareGPT/OpenAI）将样本转换为 messages，
    再用 tokenizer（ModelTokenizer）完成 tokenization，保持与 SFTDataset 一致的输出字典键。
    """

    def __init__(
        self,
        *,
        tokenizer: ModelTokenizer,
        source: str,
        message_transform: Transform,
        split: str = "train",
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        audio_text_column: str = "text",
        motion_text_column: str = "text",
        repeat_factor: int = 200,
        cutoff_audio_length: int = 200,
        cutoff_motion_length: int = 30,
        audio_stride: int = 20,
        motion_stride: int = 3,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = tokenizer

        self._data = load_dataset(source, split=split, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

        self.audio_text_column = audio_text_column
        self.motion_text_column = motion_text_column
        self.repeat_factor = repeat_factor
        self.cutoff_audio_length = cutoff_audio_length
        self.cutoff_motion_length = cutoff_motion_length
        self.audio_stride = audio_stride
        self.motion_stride = motion_stride

        self.pair_num = len(self._data) // 2
        if self.pair_num == 0:
            raise ValueError("数据量不足，无法形成语音-动作配对。")

        # 与 SFTDataset 相同的预处理流水线：message_transform -> model_transform
        self._prepare_sample = SFTTransform(
            message_transform=self._message_transform,
            model_transform=self._model_transform,
        )

    def __len__(self) -> int:
        # 防止因为数据集太小了导致一次迭代不了几次
        return len(self._data) * self.repeat_factor

    def random_drop(self, motion_cutoff_prompt_set):
        motion_cutoff_prompt_set = list(motion_cutoff_prompt_set)
        min_percent = 10
        max_percent = 20
        remove_percent = random.uniform(min_percent, max_percent) / 100
        num_to_remove = int(len(motion_cutoff_prompt_set) * remove_percent)

        indices_to_remove = random.sample(range(len(motion_cutoff_prompt_set)), num_to_remove)
        
        # 创建新列表，排除被选中要删除的元素
        return set([x for i, x in enumerate(motion_cutoff_prompt_set) if i not in indices_to_remove])
    

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = index % self.pair_num
        sample_audio = self._data[idx]
        sample_motion = self._data[idx + self.pair_num]

        audio_tokens = sample_audio[self.audio_text_column]
        motion_tokens = sample_motion[self.motion_text_column]

        if not isinstance(audio_tokens, (list, tuple)) or not isinstance(
            motion_tokens, (list, tuple)
        ):
            raise TypeError(
                f"期望列 '{self.audio_text_column}' 与 '{self.motion_text_column}' 为 list/tuple，而不是其他类型。"
            )

        # 随机对齐裁剪（对齐到 stride）
        rand_range = max(1, len(audio_tokens) - self.cutoff_audio_length + 1)
        audio_start_idx = np.random.randint(0, rand_range)
        audio_start_idx = (audio_start_idx // self.audio_stride) * self.audio_stride
        audio_end_idx = min(audio_start_idx + self.cutoff_audio_length, len(audio_tokens))
        audio_cut = audio_tokens[audio_start_idx:audio_end_idx]

        motion_start_idx = (audio_start_idx // self.audio_stride) * self.motion_stride
        motion_end_idx = min(motion_start_idx + self.cutoff_motion_length, len(motion_tokens))
        motion_cut = motion_tokens[motion_start_idx:motion_end_idx]

        motion_cutoff_prompt_set = set(motion_cut)
        motion_cutoff_prompt_set = self.random_drop(motion_cutoff_prompt_set)

        if isinstance(self._message_transform, OpenAIToMessages):
            sample = {
                "messages": [
                    {"role": "system", "content": "".join(sorted(motion_cutoff_prompt_set))},
                    {"role": "user", "content": "".join(audio_cut)},
                    {"role": "assistant", "content": "".join(motion_cut)},
                ]
            }

        # 复用 SFTTransform：生成 tokens/mask，并自动构造 labels（带位移与忽略）
        return self._prepare_sample(sample)


def icl_dataset_mhubert_1000(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_style: str = "openai",
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    audio_text_column: str = "text",
    motion_text_column: str = "text",
    repeat_factor: int = 200,
    cutoff_audio_length: int = 200,
    cutoff_motion_length: int = 30,
    audio_stride: int = 20,
    motion_stride: int = 3,
    **load_dataset_kwargs: dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    与现有 chat_dataset 风格一致的构建器：选择 ShareGPT/OpenAI 消息格式，
    并返回可选打包的 Dataset（PackedDataset）。
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": "conversations"},
            new_system_prompt=new_system_prompt,
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": "messages"},
            new_system_prompt=new_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = ICLMHuBERT1000Dataset(
        tokenizer=tokenizer,
        source=source,
        message_transform=message_transform,
        split=split,
        filter_fn=filter_fn,
        audio_text_column=audio_text_column,
        motion_text_column=motion_text_column,
        repeat_factor=repeat_factor,
        cutoff_audio_length=cutoff_audio_length,
        cutoff_motion_length=cutoff_motion_length,
        audio_stride=audio_stride,
        motion_stride=motion_stride,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds



