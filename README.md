# TextBPN-MLOCR: Advanced Multi-Lingual Scene Text Detection

[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/somos99/TextBPN-MLOCR)

**Enhanced version of TextBPN++** for robust scene text detection across multiple languages and artistic fonts. Trained on large-scale synthetic and real-world text datasets for superior performance.

## Key Features ✨
- **Multi-Lingual Support**: Arabic, Bangla, Chinese, Japanese, Korean, Latin, Hindi
- **Artistic Text Detection**: Handles stylized and decorative fonts effectively
- **Optimized Performance**: Supports modern NVIDIA GPUs (H100/H800/H20)
- **Large-scale Training**: 
  - 1.5M+ synthetic text samples 
  - 500K+ real-world text samples

## Hardware Requirements
- **GPU**: NVIDIA H100, H800, H20 (CUDA 12.2 compatible)
- **Python**: ≥ 3.9
- **CUDA**: 12.2

## Model Download
Download pre-trained models from HuggingFace Hub:  
📦 [https://huggingface.co/somos99/TextBPN-MLOCR](https://huggingface.co/somos99/TextBPN-MLOCR)

## Installation

### From PyPI
```bash
pip install textbpn-mlocr
