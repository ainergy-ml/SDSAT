import enum
from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset


class CodeLlamaSentinel():
    prefix_str = '▁<PRE>'
    middle_str = '▁<MID>'
    suffix_str = '▁<SUF>'
    eot_str = '▁<EOT>'


mdict = {
    "codellama": CodeLlamaSentinel,
}


def format_infilling(prefix, suffix, mtype, ptype):
    """ format the prompt of infilling test
    Args:
        mtype: model type
        ptype: infilling type
    """
    mobj = mdict.get(mtype, None)
    assert mobj is not None, f"Model {mtype} not found"
    if ptype == "SPM":
        prompt = mobj.prefix_str + mobj.suffix_str + suffix + mobj.middle_str + prefix
    elif ptype == "PSM":
        prompt = mobj.prefix_str + prefix + mobj.suffix_str + suffix + mobj.middle_str
    return prompt


class CodeDataset(Dataset):

    def __init__(
        self,
        name,
        dataset,
        tokenizer,
        limit,
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataset = self.dataset[: int(limit * len(self.dataset))]

    def collate_fn(self, batch):
        batch_source = [b[0] for b in batch]
        batch_target = [b[1] for b in batch]

        encoded_source = self.tokenizer(
            batch_source,
            return_tensors="pt",
        )
        encoded_target = self.tokenizer(
            batch_target,
            return_tensors="pt",
        )

        return {
            "source": {
                "input_ids": encoded_source["input_ids"],
                "attention_mask": encoded_source["attention_mask"],
                "sentences": batch_source,
            },
            "target": {
                "input_ids": encoded_target["input_ids"],
                "attention_mask": encoded_target["attention_mask"],
                "sentences": batch_target,
            },
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        sample = self.dataset[idx]
        source = sample[1]
        target = sample[2]
        return source, target



def get_humaneval_data():
    data = load_dataset("openai_humaneval")
    prompts = data['test']['prompt']
    ans = data['test']['canonical_solution']
    ids = data['test']['task_id']
    res = []
    for id, p, a in zip(ids, prompts, ans):
        res.append((id.split('/')[-1], p, a))
    return res


def get_multiple_infilling_data(mtype, ptype):
    data = load_dataset("bigcode/santacoder-fim-task")
    prefixs = data['train']['prompt']
    suffixs = data['train']['suffix']
    ans = data['train']['canonical_solution']
    lang = data['train']['language']
    res = []
    for i, (prefix, suffix, an) in enumerate(zip(prefixs, suffixs, ans)):
        inputs = format_infilling(prefix, suffix, mtype, ptype)
        res.append((i, inputs, an))
    return res

