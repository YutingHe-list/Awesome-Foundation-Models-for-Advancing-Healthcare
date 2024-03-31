# Awesome-Foundation-Models-for-Advancing-Healthcare

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo is a collection of AWESOME things about Foundation models in healthcare, including language foundation models (LFMs), vision foundation models (VFMs), bioinformatics foundation models (BFMs), and multimodal foundation models (MFMs). Feel free to star and fork.

# Contents
- [Awesome-Foundation-Models-for-Advancing-Healthcare](#Awesome-Foundation-Models-for-Advancing-Healthcare)
- [Related Survery](#related-survey)
- [Methods](#methods)
  - [LFM methods](#lfm-methods)
  - [VFM methods](#vfm-methods)
  - [BFM methods](#bfm-methods)
  - [MFM methods](#mfm-methods)
- [Datasets](#datasets)
- - [LFM data](#lfm-data)
  - [VFM data](#vfm-data)
  - [BFM data](#bfm-data)
  - [MFM data](#mfm-data)
- [Lectures and Tutorials](#lectures-and-tutorials)
- [Other Resources](#other-resources)

# Related Survey

# Methods
## LFM methods in healthcare
**2024**
- [AAAI] Zhongjing: Enhancing the chinese medical capabilities of large language model through expert feedback and realworld multi-turn dialogue. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29907) [[Code]](https://github.com/SupritYoung/Zhongjing)

- [arXiv] BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains [[Paper]](https://arxiv.org/abs/2402.10373) [[Code]](https://huggingface.co/BioMistral/BioMistral-7B)

**2023**
- [Bioinformatics] MedCPT: A method for zero-shot biomedical information retrieval using contrastive learning with PubMedBERT. [[Paper]](https://academic.oup.com/bioinformatics/article-abstract/39/11/btad651/7335842) [[Code]](https://github.com/ncbi/MedCPT)

- [arXiv] Pmc-llama: Towards building open-source language models for medicine. [[Paper]](https://arxiv.org/abs/2304.14454) [[Code]](https://github.com/chaoyi-wu/PMC-LLaMA)

- [arXiv] Meditron-70b: Scaling medical pretraining for large language models. [[Paper]](https://arxiv.org/abs/2311.16079) [[Code]](https://github.com/epfLLM/meditron)

- [arXiv] Qilin-med: Multi-stage knowledge injection advanced medical large language model. [[Paper]](https://arxiv.org/abs/2310.09089) [[Code]](https://github.com/epfLLM/meditron)

- [arXiv] Huatuogpt-ii, one-stage training for medical adaption of llms. [[Paper]](https://arxiv.org/abs/2311.09774) [[Code]](https://github.com/freedomintelligence/huatuogpt-ii)

- [NPJ Digit. Med.] A study of generative large language model for medical research and healthcare. [[Paper]](https://www.nature.com/articles/s41746-023-00958-w) [[Code]](https://github.com/uf-hobi-informatics-lab/gatortrongpt)

- [arXiv] From beginner to expert: Modeling medical knowledge into general llms. [[Paper]](https://arxiv.org/abs/2312.01040)

- [arXiv] Huatuo: Tuning llama model with chinese medical knowledge. [[Paper]](https://arxiv.org/abs/2304.06975) [[Code]](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

- [arXiv] Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge. [[Paper]](https://arxiv.org/abs/2303.14070) [[Code]](https://github.com/Kent0n-Li/ChatDoctor)

- [arXiv] Medalpaca–an open-source collection of medical conversational ai models and training data. [[Paper]](https://arxiv.org/abs/2304.08247) [[Code]](https://github.com/kbressem/medAlpaca)

- [arXiv] Alpacare: Instruction-tuned large language models for medical application. [[Paper]](https://arxiv.org/abs/2310.14558) [[Code]](https://github.com/xzhang97666/alpacare)

- [arXiv] Huatuogpt, towards taming language model to be a doctor. [[Paper]](https://arxiv.org/abs/2305.15075) [[Code]](https://github.com/FreedomIntelligence/HuatuoGPT)

- [arXiv] Doctorglm: Fine-tuning your chinese doctor is not a herculean task. [[Paper]](https://arxiv.org/abs/2304.01097) [[Code]](https://github.com/xionghonglin/DoctorGLM)

- [arXiv] Bianque: Balancing the questioning and suggestion ability of health llms with multi-turn health conversations polished by chatgpt. [[Paper]](https://arxiv.org/abs/2310.15896) [[Code]](https://github.com/scutcyr/BianQue)

- [arXiv] Taiyi: A bilingual fine-tuned large language model for diverse biomedical tasks. [[Paper]](https://arxiv.org/abs/2311.11608) [[Code]](https://github.com/DUTIR-BioNLP/Taiyi-LLM)

- [Github] Visual med-alpaca: A parameter-efficient biomedical llm with visual capabilities. [[Code]](https://github.com/cambridgeltl/visual-med-alpaca)

- [arXiv] Ophglm: Training an ophthalmology large languageand-vision assistant based on instructions and dialogue. [[Paper]](https://arxiv.org/abs/2306.12174) [[Code]](https://github.com/ML-AILab/OphGLM)

- [arXiv] Chatcad: Interactive computer-aided diagnosis on medical image using large language models. [[Paper]](https://arxiv.org/abs/2302.07257) [[Code]](https://github.com/zhaozh10/ChatCAD)

- [arXiv] Chatcad+: Towards a universal and reliable interactive cad using llms. [[Paper]](https://arxiv.org/abs/2305.15964) [[Code]](https://github.com/zhaozh10/ChatCAD)

- [arXiv] Deid-gpt: Zero-shot medical text de-identification by gpt-4. [[Paper]](https://arxiv.org/abs/2303.11032) [[Code]](https://github.com/yhydhx/ChatGPT-API)

- [arXiv] Can generalist foundation models outcompete special-purpose tuning? case study in medicine. [[Paper]](https://arxiv.org/abs/2311.16452) [[Code]](https://github.com/microsoft/promptbase)

- [arXiv] Medagents: Large language models as collaborators for zero-shot medical reasoning. [[Paper]](https://arxiv.org/abs/2311.10537) [[Code]](https://github.com/gersteinlab/MedAgents)

- [AIME] Soft-prompt tuning to predict lung cancer using primary care free-text dutch medical notes. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-34344-5_23) [[Code]](https://bitbucket.org/aumc-kik/prompt_tuning_cancer_prediction/src/master/)

- [arXiv] Clinical decision transformer: Intended treatment recommendation through goal prompting. [[Paper]](https://arxiv.org/abs/2302.00612) [[Code]](https://clinical-decision-transformer.github.io/)

- [Nature] Large language models encode clinical knowledge [[Paper]](https://www.nature.com/articles/s41586-023-06291-2) 

- [arXiv] Towards expert-level medical question answering with large language models [[Paper]](https://arxiv.org/abs/2305.09617) 

- [arXiv] Gpt-doctor: Customizing large language models for medical consultation [[Paper]](https://arxiv.org/abs/2312.10225) 

- [arXiv] Clinicalgpt: Large language models finetuned with diverse medical data and comprehensive evaluation [[Paper]](https://arxiv.org/abs/2306.09968) 

- [arXiv] Leveraging a medical knowledge graph into large language models for diagnosis prediction [[Paper]](https://arxiv.org/abs/2308.14321) 


**2022**
- [NPJ Digit. Med.] A large language model for electronic health records. [[Paper]](https://www.nature.com/articles/s41746-022-00742-2) [[Code]](https://github.com/uf-hobi-informatics-lab/GatorTron)

- [AMIA Annu. Symp. Proc.] Healthprompt: A zero-shot learning paradigm for clinical natural language processing. [[Paper]](https://pubmed.ncbi.nlm.nih.gov/37128372/)

- [BioNLP] Position-based prompting for health outcome generation [[Paper]](https://aclanthology.org/2022.bionlp-1.3/) 


**2021**
- [ACM Trans. Comput. Healthc.] Domain-specific language model pretraining for biomedical natural language processing. [[Paper]](https://dl.acm.org/doi/10.1145/3458754) [[Code]](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)

**2020**
- [JMIR Med. Info.] Modified bidirectional encoder representations from transformers extractive summarization model for hospital information systems based on character-level tokens (alphabert): development and performance evaluation. [[Paper]](https://medinform.jmir.org/2020/4/e17787/) [[Code]](https://github.com/wicebing/AlphaBERT)

- [Scientific reports] Behrt: transformer for electronic health records. [[Paper]](https://www.nature.com/articles/s41598-020-62922-y) [[Code]](https://github.com/deepmedicine/BEHRT)

- [BioNLP] BioBART: Pretraining and evaluation of a biomedical generative language model. [[Paper]](https://aclanthology.org/2022.bionlp-1.9/) [[Code]](https://github.com/GanjinZero/BioBART)

- [Method. Biochem. Anal.] Biobert: a pre-trained biomedical language representation model for biomedical text mining. [[Paper]](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) [[Code]](https://github.com/dmis-lab/biobert)

**2019**
- [NPJ Digit. Med.] ClinicalBERT: A hybrid learning model for natural language inference in healthcare using BERT. [[Paper]](https://www.nature.com/articles/s41746-022-00742-2) [[Code]](https://github.com/EmilyAlsentzer/clinicalBERT)

## VFM methods
## BFM methods
**2024**
  - [Nucleic Acids Research] Multiple sequence alignment-based RNA language model and its application to structural inference. [[Paper]](https://academic.oup.com/nar/article/52/1/e3/7369930), [[Code]](https://github.com/yikunpku/RNA-MSM)
  - [Nature Methods] scGPT: toward building a foundation model for single-cell multi-omics using generative AI. [[Paper]](https://www.nature.com/articles/s41592-024-02201-0), [[Code]](https://github.com/bowang-lab/scGPT)


**2023**
  - [arXiv] DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks. [[Paper]](https://arxiv.org/abs/2307.05628), [[Code]](https://github.com/TencentAILabHealthcare/DNAGPT)
  - [arXiv] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. [[Paper]](https://arxiv.org/abs/2306.15794), [[Code]](https://github.com/HazyResearch/hyena-dna)
  - [Nature Biotechnology] Large language models generate functional protein sequences across diverse families. [[Paper]](https://www.nature.com/articles/s41587-022-01618-2), [[Code]](https://github.com/salesforce/progen)
  - [Cell Systems] ProGen2: Exploring the boundaries of protein language models. [[Paper]](https://www.cell.com/cell-systems/abstract/S2405-4712(23)00272-7), [[Code]](https://github.com/salesforce/progen/tree/main/progen2)
  - [Nature] Transfer learning enables predictions in network biology. [[Paper]](https://www.nature.com/articles/s41586-023-06139-9), [[Code]](https://huggingface.co/ctheodoris/Geneformer)
  - [arXiv] DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. [[Paper]](https://arxiv.org/abs/2306.15006), [[Code]](https://github.com/Zhihan1996/DNABERT_2)
  - [bioRxiv] The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1), [[Code]](https://github.com/instadeepai/nucleotide-transformer)
  - [bioRxiv] GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v2), [[Code]](https://github.com/AIRI-Institute/GENA_LM)
  - [bioRxiv] Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.01.31.526427v2), [[Code]](https://github.com/biomed-AI/SpliceBERT)
  - [bioRxiv] A 5’ UTR Language Model for Decoding Untranslated Regions of mRNA and Function Predictions. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.10.11.561938v1.full), [[Code]](https://github.com/a96123155/UTR-LM)
  - [bioRxiv] Deciphering 3’ UTR mediated gene regulation using interpretable deep representation learning. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.09.08.556883v1), [[Code]](https://github.com/yangyn533/3UTRBERT)
  - [Science] Evolutionary-scale prediction of atomic-level protein structure with a language model. [[Paper]](https://www.science.org/doi/10.1126/science.ade2574), [[Code]](https://github.com/facebookresearch/esm)
  - [bioRxiv] Universal Cell Embeddings: A Foundation Model for Cell Biology. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1), [[Code]](https://github.com/snap-stanford/UCE)
  - [bioRxiv] Large Scale Foundation Model on Single-cell Transcriptomics. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705v4), [[Code]](https://github.com/biomap-research/scFoundation)

  - [arXiv] Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning. [[Paper]](https://arxiv.org/abs/2306.04371), [[Code]](https://github.com/PharMolix/OpenBioMed)

  - [bioRxiv] CodonBERT: Large Language Models for mRNA design and optimization. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v2), [[Code]](https://github.com/Sanofi-Public/CodonBERT)
  - [bioRxiv] xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v4)
  - [bioRxiv] GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.10.16.562533v2), [[Code]](https://github.com/yiqunchen/GenePT)


**2022**
  - [Nature Machine Intelligence] scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. [[Paper]](https://www.nature.com/articles/s42256-022-00534-z), [[Code]](https://github.com/TencentAILabHealthcare/scBERT)
  - [bioRxiv] Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions. [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503062v2), [[Code]](https://github.com/ml4bio/RNA-FM)
  - [NAR Genomics & Bioinformatics] Informative RNA base embedding for RNA structural alignment and clustering by deep representation learning. [[Paper]](https://academic.oup.com/nargab/article/4/1/lqac012/6534363), [[Code]](https://github.com/mana438/RNABERT)
  - [Nature Biotechnology] Single-sequence protein structure prediction using language models and deep learning. [[Paper]](https://www.nature.com/articles/s41587-022-01432-w), [[Code]](https://github.com/aqlaboratory/rgn2)


**2021**
  - [Bioinformatics] DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. [[Paper]](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680), [[Code]](https://github.com/jerryji1993/DNABERT)
  - [IEEE TPAMI] ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. [[Paper]](https://ieeexplore.ieee.org/document/9477085), [[Code]](https://github.com/agemagician/ProtTrans)
  - [ICML 2021] MSA Transformer. [[Paper]](https://proceedings.mlr.press/v139/rao21a.html), [[Code]](https://github.com/rmrao/msa-transformer)
  - [PNAS] Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. [[Paper]](https://www.pnas.org/doi/10.1073/pnas.2016239118), [[Code]](https://github.com/facebookresearch/esm)
  - [Nature] Highly accurate protein structure prediction with AlphaFold. [[Paper]](https://www.nature.com/articles/s41586-021-03819-2), [[Code]](https://github.com/google-deepmind/alphafold)
  - [arXiv] Multi-modal Self-supervised Pre-training for Regulatory Genome Across Cell Types. [[Paper]](https://arxiv.org/abs/2110.05231), [[Code]](https://github.com/ZovcIfzm/GeneBERT)



## MFM methods
**2024**
- [arXiv] BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. [[Paper]](https://arxiv.org/pdf/2303.00915.pdf) [[Code]](https://github.com/LightersWang/BiomedCLIP-LoRA)
- [ICASSP] Etp: Learning transferable ecg representations via ecg-text pretraining. [[Paper]](https://ieeexplore.ieee.org/document/10446742)
- [NeurIPS] Med-unic: Unifying cross-lingual medical vision language pre-training by diminishing bias. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/af38fb8e90d586f209235c94119ba193-Paper-Conference.pdf) [[Code]](https://github.com/SUSTechBruce/Med-UniC)
- [NeurIPS] Quilt-1m: One million image-text pairs for histopathology. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/775ec578876fa6812c062644964b9870-Paper-Datasets_and_Benchmarks.pdf) [[Code]](https://github.com/wisdomikezogwo/quilt1m)


**2023**
- [ICLR] Advancing radiograph representation learning with masked record modeling. [[Paper]](https://openreview.net/forum?id=w-x7U26GM7j) [[Code]](https://github.com/RL4M/MRM-pytorch)
- [arXiv] BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks. [[Paper]](https://arxiv.org/abs/2305.17100) [[Code]](https://github.com/taokz/BiomedGPT)
- [arXiv] Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data. [[Paper]](https://arxiv.org/pdf/2308.02463.pdf) [[Code]](https://github.com/chaoyi-wu/RadFM)
- [CVPR] Visual language pretrained multiple instance zero-shot  transfer for histopathology images. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.pdf) [[Code]](https://github.com/mahmoodlab/MI-Zero)
- [ICCV] Medklip: Medical knowledge enhanced language-image pre-training. [[Paper]](https://chaoyi-wu.github.io/MedKLIP/) [[Code]](https://github.com/MediaBrain-SJTU/MedKLIP)
- [arXiv] UniBrain: Universal Brain MRI Diagnosis with Hierarchical Knowledge-enhanced Pre-training. [[Paper]](https://arxiv.org/abs/2309.06828) [[Code]](https://github.com/ljy19970415/UniBrain)
- [EACL] PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain. [[Paper]](https://aclanthology.org/2023.findings-eacl.88.pdf) [[Code]](https://github.com/sarahESL/PubMedCLIP)
- [MICCAI] M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_61) [[Code]](https://github.com/cheliu-computation/M-FLAG-MICCAI2023)
- [arXiv] IMITATE: Clinical Prior Guided Hierarchical Vision-Language Pre-training. [[Paper]](https://arxiv.org/abs/2310.07355)
- [arXiv] CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training. [[Paper]](https://arxiv.org/abs/2310.13292) [[Code]](https://github.com/kakaobrain/cxr-clip)
- [BIBM] UMCL: Unified Medical Image-Text-Label Contrastive Learning With Continuous Prompt. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10386034)
- [Nature Communications] Knowledge-enhanced visual-language pre-training on chest radiology images. [[Paper]](https://www.nature.com/articles/s41467-023-40260-7)
- [Nature Machine Intelligence] Multi-modal molecule structure–text model for text-based retrieval and editing. [[Paper]](https://www.nature.com/articles/s42256-023-00759-6) [[Code]](https://github.com/chao1224/MoleculeSTM/tree/main)
- [MICCAI] Clip-lung: Textual knowledge-guided lung nodule malignancy prediction. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_38)
- [MICCAI] Breaking with fixed set pathology recognition through report-guided contrastive training. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_66)
- [MICCAI] Pmc-clip: Contrastive language-image pre-training using biomedical documents. [[Paper]](https://arxiv.org/pdf/2303.07240.pdf) [[Code]](https://github.com/WeixiongLin/PMC-CLIP)
- [arXiv] Enhancing representation in radiography-reports foundation model: A granular alignment algorithm using masked contrastive learning. [[Paper]](https://arxiv.org/abs/2309.05904) [[Code]](https://github.com/SZUHvern/MaCo)
- [ICCV] Prior: Prototype representation joint learning from  medical images and reports. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_PRIOR_Prototype_Representation_Joint_Learning_from_Medical_Images_and_Reports_ICCV_2023_paper.html) [[Code]](https://github.com/QtacierP/PRIOR)


**2022**
- [JMLR] Contrastive learning of medical visual representations from paired images and text. [[Paper]](https://proceedings.mlr.press/v182/zhang22a/zhang22a.pdf) [[Code]](https://github.com/edreisMD/ConVIRT-pytorch)
- [ECCV] Joint learning of localized representations from medical images and reports. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_39)
- [NeurIPS] Multi-granularity cross-modal alignment for generalized medical visual representation learning. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/d925bda407ada0df3190df323a212661-Paper-Conference.pdf) [[Code]](https://github.com/HKU-MedAI/MGCA)
- [AAAI] Clinical-BERT: Vision-Language Pre-training for Radiograph Diagnosis and Reports Generation.  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20204)
- [MICCAI] Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training. [[Paper]](https://arxiv.org/abs/2209.07098) [[Code]](https://github.com/zhjohnchan/M3AE)
- [JBHI] Multi-modal understanding and generation for medical images and text via vision-language pre-training. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9894658) [[Code]](https://github.com/SuperSupermoon/MedViLLV)
- [ACM MM] Align, reason and learn: Enhancing medical vision-and-language pre-training with knowledge. [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3547948) [[Code]](https://github.com/zhjohnchan/ARL)
- [ECCV] Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20059-5_1)


**2021**
- [arXiv] MMBERT: Multimodal BERT Pretraining for Improved Medical VQA. [[Paper]](https://arxiv.org/abs/2104.01394) [[Code]](https://github.com/VirajBagal/MMBERT)
- [ICCV] GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-Efficient Medical Image Recognition. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html) [[Code]](https://github.com/marshuang80/gloria)






# Datasets
## LFM data
|                           Dataset  Name                               | Text Types  |            Scale           |    Task    |                       Link                             |
| :-------------------------------------------------------------------: | :-------: | :------------------------: | :--------: | :----------------------------------------------------: |
|[PubMed](https://pubmed.ncbi.nlm.nih.gov/download/) | Literature | 18B tokens |  Language modeling |[*](https://pubmed.ncbi.nlm.nih.gov/download/)|
|[MedC-I](https://arxiv.org/abs/2304.14454)| Literature | 79.2B tokens |  Dialogue |[*](https://huggingface.co/datasets/axiong/pmc_llama_instructions)|
|[Guidelines](https://arxiv.org/abs/2311.16079)| Literature | 47K instances |  Language modeling |[*](https://huggingface.co/datasets/epfl-llm/guidelines)|
|[PMC-Patients](https://www.nature.com/articles/s41597-023-02814-8) | Literature | 167K instances |  Information retrieval |[*](https://github.com/pmc-patients/pmc-patients)|
|[MIMIC-III](https://www.nature.com/articles/sdata201635) | Health records | 122K instances |  Language modeling |[*](https://physionet.org/content/mimiciii/1.4/)|
|[MIMIC-IV](https://www.nature.com/articles/s41597-022-01899-x)| Health record | 299K instances |  Language modeling |[*](https://physionet.org/content/mimiciv/2.2/)|
|[eICU-CRDv2.0](https://www.nature.com/articles/sdata2018178)| Health record |  200K instances  |  Language modeling |[*](https://physionet.org/content/eicu-crd/2.0/)|
|[EHRs](https://www.nature.com/articles/s41746-022-00742-2)| Health record | 82B tokens |  Named entity recognition, Relation extraction, Semantic textual similarity, Natural language inference, Dialogue | - |
|[MD-HER](https://arxiv.org/abs/2306.09968)| Health record | 96K instances	 |  Dialogue, Question answering | - |
|[IMCS-21](https://academic.oup.com/bioinformatics/article/39/1/btac817/6947983)| Dialogue | 4K instances |  Dialogue |[*](https://github.com/lemuria-wchen/imcs21)|
|[Huatuo-26M](https://arxiv.org/abs/2305.01526)| Dialogue | 26M instances |  Question answering |[*](https://github.com/FreedomIntelligence/Huatuo-26M)|
|[MedInstruct-52k](https://arxiv.org/abs/2310.14558)| Dialogue | 52K instances |  Dialogue |[*](https://github.com/XZhang97666/AlpaCare)|
|[MASH-QA](https://aclanthology.org/2020.findings-emnlp.342/)| Dialogue | 35K instances |  Dialogue |[*](https://github.com/mingzhu0527/MASHQA)|
|[MedQuAD](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4)| Dialogue | 47K instances |  Dialogue |[*](https://github.com/abachaa/MedQuAD)|
|[MedDG](https://aclanthology.org/2020.emnlp-main.743/)| Dialogue | 17K instances |  Dialogue |[*](https://github.com/lwgkzl/MedDG)|
|[CMExam](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html)| Dialogue | 68K instances |  Dialogue |[*](https://github.com/williamliujl/CMExam)|
|[cMedQA2](https://ieeexplore.ieee.org/document/8548603/)| Dialogue | 108K instances |  Dialogue |[*](https://github.com/zhangsheng93/cMedQA2)|
|[CMtMedQA](https://ojs.aaai.org/index.php/AAAI/article/view/29907)| Dialogue | 70K instances |  Dialogue |[*](https://github.com/SupritYoung/Zhongjing)|
|[CliCR](https://aclanthology.org/N18-1140/)| Dialogue | 100K instances |  Dialogue |[*](https://github.com/clips/clicr)|
|[webMedQA](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0761-8)| Dialogue | 63K instances |  Dialogue |[*](https://github.com/hejunqing/webMedQA)|
|[ChiMed](https://arxiv.org/abs/2310.09089)| Dialogue | 1.59B tokens |  Dialogue |[*](https://huggingface.co/datasets/williamliu/ChiMed/tree/main)|
|[MedDialog](https://aclanthology.org/2020.emnlp-main.743/)| Dialogue | 20K instances |  Dialogue |[*](https://github.com/UCSD-AI4H/Medical-Dialogue-System)|
|[CMD](https://github.com/Toyhom/Chinese-medical-dialogue-data)| Dialogue | 882K instances |  Dialogue |[*](https://github.com/Toyhom/Chinese-medical-dialogue-data)|
|[BianqueCorpus](https://arxiv.org/abs/2310.15896)| Dialogue | 2.4M instances |  Dialogue |[*](https://github.com/scutcyr/BianQue)|
|[MedQA](https://www.mdpi.com/2076-3417/11/14/6421)| Dialogue | 4K instances |  Dialogue |[*](https://github.com/jind11/MedQA)|
|[HealthcareMagic](https://huggingface.co/datasets/RafaelMPereira/HealthCareMagic-100k-Chat-Format-en)| Dialogue | 100K instances |  Dialogue |[*](https://huggingface.co/datasets/RafaelMPereira/HealthCareMagic-100k-Chat-Format-en)|
|[iCliniq](https://drive.google.com/file/d/1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA/view)| Dialogue | 10K instances |  Dialogue |[*](https://drive.google.com/file/d/1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA/view)|
|[CMeKG-8K](https://www.mdpi.com/2078-2489/11/4/186)| Dialogue | 8K instances |  Dialogue |[*](https://github.com/WENGSYX/CMKG)|
|[Hybrid SFT](https://aclanthology.org/2023.findings-emnlp.725/)| Dialogue | 226K instances |  Dialogue |[*](https://github.com/FreedomIntelligence/HuatuoGPT)|
|[VariousMedQA](https://github.com/cambridgeltl/visual-med-alpaca)| Dialogue | 54K instances |  Dialogue |[*](https://github.com/cambridgeltl/visual-med-alpaca)|
|[Medical Meadow](https://arxiv.org/abs/2304.08247)| Dialogue | 160K instances |  Dialogue |[*](https://github.com/kbressem/medAlpaca)|
|[MultiMedQA](https://arxiv.org/abs/2305.09617)| Dialogue | 193K instances |  Dialogue | - |

## VFM data
|                           Dataset  Name                               | Modality  |            Scale           |    Task    |                       Link                             |
| :-------------------------------------------------------------------: | :-------: | :------------------------: | :--------: | :----------------------------------------------------: |
|[LIMUC](https://academic.oup.com/ibdjournal/article/29/9/1431/6830946) | Endoscopy | 1043 videos (11276 frames) |  Detection |[*](https://zenodo.org/records/5827695#.Yi8GJ3pByUk)|
|[SUN](https://www.sciencedirect.com/science/article/pii/S0016510720346551)| Endoscopy | 1018 videos (158,690 frames) |  Detection |[*](http://amed8k.sundatabase.org/)|
|[Kvasir-Capsule](https://www.nature.com/articles/s41597-021-00920-z)| Endoscopy | 117 videos (4,741,504 frames) |  Detection |[*](https://datasets.simula.no/kvasir-capsule/)|
|[EndoSLAM](https://www.sciencedirect.com/science/article/pii/S1361841521001043) | Endoscopy | 1020 videos (158,690 frames) |  Detection, Registration |[*](https://github.com/CapsuleEndoscope/EndoSLAM)|
|[LDPolypVideo](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_37) | Endoscopy | 263 videos (895,284 frames) |  Detection |[*](https://github.com/dashishi/LDPolypVideo-Benchmark)|
|[HyperKvasir](https://www.nature.com/articles/s41597-020-00622-y)| Endoscopy | 374 videos (1,059,519 frames) |  Detection |[*](https://datasets.simula.no/hyper-kvasir)|
|[CholecT45](https://arxiv.org/pdf/2204.05235.pdf)| Endoscopy |  45 videos (90489 frames)  |  Segmentation, Detection |[*](https://github.com/CAMMA-public/cholect45)|
|[DeepLesion]()| CT slices (2D) | 32,735 images |  Segmentation, Registration |[*](nihcc.app.box.com)|
|[]()| Endoscopy | 1,018 volumes	 |  Detection |[*]()|
|[   ]()| Endoscopy |  |  Detection |[*]()|
|[   ]()| Endoscopy |  |  Detection |[*]()|



## BFM data
|                           Dataset  Name                               | Modality  |            Scale           |    Task    |                       Link                             |
| :-------------------------------------------------------------------: | :-------: | :------------------------: | :--------: | :----------------------------------------------------: |
|[CellxGene Corpus]([https://academic.oup.com/ibdjournal/article/29/9/1431/6830946](https://www.biorxiv.org/content/10.1101/2023.10.30.563174v1)) | scRNA-seq | over 72M scRNA-seq data |  Single cell omics study |[*](https://cellxgene.cziscience.com)|
|[NCBI GenBank](https://doi.org/10.1093/nar/gks1195)| DNA | 3.7B sequences |  Genomics study |[*](https://www.ncbi.nlm.nih.gov/genbank/)|
|[SCP](https://www.biorxiv.org/content/10.1101/2023.07.13.548886v1)| scRNA-seq | over 40M scRNA-seq data | Single cell omics study |[*](https://singlecell.broadinstitute.org/single_cell)|
|[Gencode](https://doi.org/10.1093/nar/gky955)| DNA |  |  Genomics study |[*](https://www.gencodegenes.org/)|
|10x Genomics| scRNA-seq, DNA |  |  Single cell omics and genomics study |[*](https://support.10xgenomics.com/single-cell-gene-expression/datasets)|
|ABC Atlas| scRNA-seq | over 15M scRNA-seq data | Single cell omics study |[*](https://portal.brain-map.org)|
|[Human Cell Atlas](https://doi.org/10.7554/eLife.27041)| scRNA-seq | over 50M scRNA-seq data | Single cell omics study |[*](https://www.humancellatlas.org/)|
|[UCSC Genome Browser](https://doi.org/10.1093/nar/gkad987)| DNA |  |  Genomics study |[*](https://hgdownload.soe.ucsc.edu/downloads.html)|
|[Ensembl Project](https://academic.oup.com/nar/article/51/D1/D933/6786199)| Protein |  |  Proteomics study |[*](https://ensembl.org/index.html)|
|[RNAcentral database](https://doi.org/10.1093/nar/gky1034)| RNA | 36M sequences |  Transcriptomics study |[*](https://rfam.org)|
|[AlphaFold DB](https://www.nature.com/articles/s41586-021-03819-2)| Protein | 214M structures |  Proteomics study |[*](https://alphafold.ebi.ac.uk/download)|
|[PDBe](https://europepmc.org/articles/PMC7145656)| Protein |  |  Proteomics study |[*](https://www.ebi.ac.uk/pdbe/)|
|[UniProt](https://doi.org/10.1093/nar/gkac1052)| Protein | over 250M sequences |  Proteomics study |[*](https://www.uniprot.org)|
|[LINCS L1000](http://lincsportal.ccs.miami.edu/datasets/#/view/LDS-1398)| Small molecules | 1,000 genes with 41k small molecules |  Disease research, drug response |[*](https://lincsportal.ccs.miami.edu/dcic-portal/)|
|[GDSC](https://doi.org/10.1093/nar/gks1111)| Small molecules | 1,000 cancer cells with 400 compounds |  Disease research, drug response |[*](https://www.cancerrxgene.org)|
|[CCLE](https://doi.org/10.1038/s41586-019-1186-3)|  |  |  Bioinformatics study |[*](https://sites.broadinstitute.org/ccle/datasets)|
|[TCGA]()|  |  |  Bioinformatics study |[*](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)|
|[CGGA](https://www.sciencedirect.com/science/article/pii/S1672022921000450)|  |  |  Bioinformatics study |[*](http://www.cgga.org.cn)|
|[UK Biobank](https://www.nature.com/articles/s41586-018-0579-z)|  |  |  Bioinformatics study |[*](https://www.ukbiobank.ac.uk)|


## MFM data

# Lectures and Tutorials

[Bioinformatics - Geneformer Tutorial (unofficial)](https://wang-lab.hkust.edu.hk/others/Tutorials/geneformer/readme.html)

# Other Resources



