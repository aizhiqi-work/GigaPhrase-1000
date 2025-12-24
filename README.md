# LibriPhrase & GigaPhrase Phrase-Level ASR Datasets

本仓库汇集并发布 **LibriPhrase** 与 **GigaPhrase** 系列开源数据集，以及配套的数据处理脚本，用于将 **ASR 级别的数据高效转换为 Phrase（短语）级别数据**，面向 **Keyword Spotting (KWS)**、**User-Defined KWS** 与 **短语级语音理解** 等研究方向。

---

## 📦 数据集概览

### 🔹 LibriPhrase 系列

基于 LibriSpeech 构建的高质量短语级数据集，覆盖多规模设置：

* **LibriPhrase-100**

  * Anchors 数量：**12k**
  * HuggingFace：👉 `ZhiqiAi/LibriPhrase-100`

* **LibriPhrase-460**

  * Anchors 数量：**78k**
  * HuggingFace：👉 `ZhiqiAi/LibriPhrase-460`

---

### 🔹 GigaPhrase 系列

在更大规模语音语料上构建的超大规模 Phrase 数据集，用于数据规模扩展与鲁棒性研究：

* **GigaPhrase-1000**

  * Anchors 数量：**155k**
  * 包含：**LibriPhrase-460 (LP-460)**
  * HuggingFace：👉 `ZhiqiAi/GigaPhrase-1000`

---

## 🛠 数据处理脚本

本仓库同时提供 **LibriPhrase** 与 **GigaPhrase** 的完整数据处理脚本，支持：

* 从 **ASR 级别数据** 自动构建 Phrase 级样本
* 高效生成 **phrase anchors**
* 支持大规模音频数据并行处理
* 适用于 Whisper / wav2vec2 / HuBERT / Conformer 等 ASR 输出

👉 目标：**显著降低 ASR → Phrase 数据构建成本，加速 KWS 相关研究与复现**。

---

## 🚀 使用场景

* User-Defined Keyword Spotting (UD-KWS)
* Phrase-level Keyword Spotting
* Two-stage / Cascaded KWS
* ASR + KWS 联合建模
* 数据规模扩展与鲁棒性分析

---

## 📚 引用（Citation）

如果你在研究或项目中使用了本数据集或脚本，请引用以下论文：

```bibtex
@article{ds-kws2024,
  title   = {Dual Data Scaling for Robust Two-Stage User-Defined Keyword Spotting},
  author  = {Zhiqi Ai et al.},
  journal = {arXiv preprint arXiv:2510.10740},
  year    = {2024}
}
```

📄 Paper: [https://arxiv.org/abs/2510.10740](https://arxiv.org/abs/2510.10740)
📝 Status: **Under Review**

---

## 📬 联系方式

如有问题、建议或合作意向，欢迎通过 HuggingFace 或 GitHub issue 联系。

---

**⭐ 如果该项目对你有帮助，欢迎 Star / Cite / Share！**
