<h1 align="center">Believe Your Eyes: Backdoor Cleaning without External Guidance in MLLM Fine-tuning</h1>

<p align="center"><em>Xuankun Rong&dagger;, Wenke Huang&dagger;, Jian Liang, Jinhe Bi, Xun Xiao, Yiming Li, Bo Du, Mang Ye*</em></p>

<div align="center">
<img alt="method" src="assets/BYE.png">
</div>

<h2> 🙌 Abstract </h2>

Multimodal Large Language Models (MLLMs) are increasingly deployed in fine-tuning-as-a-service (FTaaS) settings, where user-submitted datasets adapt general-purpose models to downstream tasks. This flexibility, however, introduces serious security risks, as malicious fine-tuning can implant backdoors into MLLMs with minimal effort. In this paper, we observe that backdoor triggers systematically disrupt cross-modal processing by causing abnormal attention concentration on non-semantic regions—a phenomenon we term **attention collapse**. Based on this insight, we propose **Believe Your Eyes (BYE)**, a data filtering framework that leverages attention entropy patterns as self-supervised signals to identify and filter backdoor samples. BYE operates via a three-stage pipeline: (1) extracting attention maps using the fine-tuned model, (2) computing entropy scores and profiling sensitive layers via bimodal separation, and (3) performing unsupervised clustering to remove suspicious samples. Unlike prior defenses, BYE requires no clean supervision, auxiliary labels, or model modifications. Extensive experiments across various datasets, models, and diverse trigger types validate BYE’s effectiveness: it achieves near-zero attack success rates while maintaining clean-task performance, offering a robust and generalizable solution against backdoor threats in MLLMs.

<h2 id="citation"> 🥳 Citation </h2>

Please kindly cite this paper in your publications if it helps your research:

```bibtex
@article{rong2025backdoor,
  title={Backdoor Cleaning without External Guidance in MLLM Fine-tuning},
  author={Rong, Xuankun and Huang, Wenke and Liang, Jian and Bi, Jinhe and Xiao, Xun and Li, Yiming and Du, Bo and Ye, Mang},
  journal={arXiv preprint arXiv:2505.16916},
  year={2025}
}
```
