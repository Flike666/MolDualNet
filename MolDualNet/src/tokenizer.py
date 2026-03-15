"""
SMILES分词器模块
SMILES Tokenizer Module
"""

import re
import json
from typing import List, Dict, Optional
from collections import Counter


class SmilesTokenizer:
    """
    SMILES字符级分词器

    支持处理双字符原子(Br, Cl等)、括号、数字等
    """

    # 特殊token
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    # SMILES正则模式 - 匹配双字符原子、单字符和其他符号
    SMILES_PATTERN = r"(\[[^\]]+\]|Br?|Cl?|Si|Se|se|@@?|[A-Z]|[a-z]|[0-9]|[#%\)\(\+\-\\\/\=\@\.\:\*\$])"

    def __init__(self, vocab_file: Optional[str] = None, max_length: int = 256):
        """
        Args:
            vocab_file: 词表文件路径（JSON格式）
            max_length: 最大序列长度
        """
        self.max_length = max_length
        self.pattern = re.compile(self.SMILES_PATTERN)

        # 初始化词表
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.MASK_TOKEN,
        ]

        if vocab_file is not None:
            self.load_vocab(vocab_file)
        else:
            self._init_default_vocab()

    def _init_default_vocab(self):
        """初始化默认词表（覆盖常见SMILES字符）"""
        # 常见SMILES token
        smiles_tokens = [
            # 原子
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 'Cl', 'Br', 'I', 'P', 'p',
            'B', 'b', 'Si', 'Se', 'se', 'H',
            # 键和结构
            '-', '=', '#', ':', '/', '\\', '@', '@@',
            # 括号和数字
            '(', ')', '[', ']',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%',
            # 电荷
            '+', '-',
            # 点（用于断开的片段）
            '.',
            # 通配符
            '*',
            # 括号内的常见模式
            '[C]', '[N]', '[O]', '[S]', '[P]', '[H]',
            '[C@]', '[C@@]', '[N@]', '[N@@]',
            '[C@H]', '[C@@H]', '[N@H]', '[N@@H]',
            '[nH]', '[NH]', '[OH]', '[SH]',
            '[Na]', '[K]', '[Ca]', '[Mg]', '[Zn]', '[Fe]', '[Cu]',
            '[Cl-]', '[Br-]', '[I-]', '[O-]', '[N-]', '[S-]',
            '[NH+]', '[NH2+]', '[NH3+]', '[O+]',
            '[Si]', '[B]', '[Se]',
        ]

        # 构建词表
        self.token2idx = {}
        self.idx2token = {}

        # 添加特殊token
        for i, token in enumerate(self.special_tokens):
            self.token2idx[token] = i
            self.idx2token[i] = token

        # 添加SMILES token
        idx = len(self.special_tokens)
        for token in smiles_tokens:
            if token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1

        self.vocab_size = len(self.token2idx)

    def tokenize(self, smiles: str) -> List[str]:
        """
        将SMILES字符串分词

        Args:
            smiles: SMILES字符串

        Returns:
            tokens: token列表
        """
        tokens = self.pattern.findall(smiles)
        return tokens

    def encode(self, smiles: str, add_special_tokens: bool = True,
               padding: bool = True, truncation: bool = True) -> Dict[str, List[int]]:
        """
        将SMILES编码为索引

        Args:
            smiles: SMILES字符串
            add_special_tokens: 是否添加[CLS]和[SEP]
            padding: 是否padding到max_length
            truncation: 是否截断到max_length

        Returns:
            dict: 包含input_ids和attention_mask
        """
        tokens = self.tokenize(smiles)

        # 转换为索引
        token_ids = []
        for token in tokens:
            if token in self.token2idx:
                token_ids.append(self.token2idx[token])
            else:
                # 尝试查找括号内的内容
                token_ids.append(self.token2idx[self.UNK_TOKEN])

        # 添加特殊token
        if add_special_tokens:
            token_ids = [self.token2idx[self.CLS_TOKEN]] + token_ids + [self.token2idx[self.SEP_TOKEN]]

        # 截断
        if truncation and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # 创建attention mask
        attention_mask = [1] * len(token_ids)

        # Padding
        if padding:
            pad_length = self.max_length - len(token_ids)
            if pad_length > 0:
                token_ids = token_ids + [self.token2idx[self.PAD_TOKEN]] * pad_length
                attention_mask = attention_mask + [0] * pad_length

        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将索引解码为SMILES字符串

        Args:
            token_ids: token索引列表
            skip_special_tokens: 是否跳过特殊token

        Returns:
            smiles: SMILES字符串
        """
        tokens = []
        for idx in token_ids:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        return ''.join(tokens)

    def batch_encode(self, smiles_list: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        """
        批量编码SMILES

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            dict: 包含input_ids和attention_mask的批次
        """
        batch_input_ids = []
        batch_attention_mask = []

        for smiles in smiles_list:
            encoded = self.encode(smiles, **kwargs)
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask
        }

    def build_vocab_from_data(self, smiles_list: List[str], min_freq: int = 1):
        """
        从数据构建词表

        Args:
            smiles_list: SMILES字符串列表
            min_freq: token的最小出现频率
        """
        # 统计所有token的频率
        token_counter = Counter()
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            token_counter.update(tokens)

        # 重建词表
        self.token2idx = {}
        self.idx2token = {}

        # 添加特殊token
        for i, token in enumerate(self.special_tokens):
            self.token2idx[token] = i
            self.idx2token[i] = token

        # 添加满足频率要求的token
        idx = len(self.special_tokens)
        for token, freq in token_counter.most_common():
            if freq >= min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1

        self.vocab_size = len(self.token2idx)
        print(f"Built vocabulary with {self.vocab_size} tokens")

    def save_vocab(self, path: str):
        """保存词表到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        """从文件加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            self.token2idx = json.load(f)
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    @property
    def pad_token_id(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def cls_token_id(self) -> int:
        return self.token2idx[self.CLS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.token2idx[self.SEP_TOKEN]


if __name__ == "__main__":
    # 测试分词器
    tokenizer = SmilesTokenizer(max_length=128)

    test_smiles = [
        "CCO",  # 乙醇
        "c1ccccc1",  # 苯
        "CC(=O)O",  # 醋酸
        "O=C(NCCC(=O)OC)[C@H](N)CC(=O)O",  # 复杂分子
    ]

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    for smiles in test_smiles:
        tokens = tokenizer.tokenize(smiles)
        encoded = tokenizer.encode(smiles)
        decoded = tokenizer.decode(encoded['input_ids'])

        print(f"\nSMILES: {smiles}")
        print(f"Tokens: {tokens}")
        print(f"Input IDs: {encoded['input_ids'][:20]}...")
        print(f"Decoded: {decoded}")
