# BLIP 与 BLIP-2 模型解析 - Google Gemini

好的，我已为您在笔记中适当地补充了模型架构与数据流向的图解，这样能更直观地理解 BLIP 和 BLIP-2 的核心设计。

以下是补充和优化后的 Markdown 笔记：


# BLIP 与 BLIP-2 多模态预训练模型深度解析

BLIP 提出了一种基于预训练的方法，通过联合训练视觉和语言模型来提升多模态任务的性能。BLIP-2 则提出了一种更简洁的预训练方法，利用现有的单模态视觉和文本预训练模型，以减少计算成本和避免灾难性遗忘问题。


## 一、 BLIP 简介

**BLIP (Bootstrapping Language-Image Pretraining)** 是 Salesforce 在 2022 年提出的多模态框架。它实现了**理解和生成的统一**，引入了跨模态的编码器和解码器，促进了跨模态信息的流动，在多项视觉和语言任务中取得了 SOTA 性能。

在 AIGC 领域中，BLIP 通常用来给图像生成提示词（Prompt），高质量的 Prompt 对交叉注意力（Cross-Attention）的微调非常关键。例如，ControlNet 中的 Automatic Prompt 功能就是由 BLIP 生成的。


> **为什么叫 Bootstrapping？**
> 因为训练数据来自网络图文对，包含大量噪声。BLIP 增加了一个在线数据打标签和清理的任务，把处理好的数据继续用来迭代原模型，这种自我提升的过程被称为 Bootstrapping（自举）。


### 1.1 模型结构

BLIP 引入了编码器-解码器的多模态混合结构 **MED（Multimodal mixture of Encoder-Decoder）**，能够有效地进行多任务预学习和迁移学习。MED 包括：

- **两个单模态编码器**：图像编码器（Image Encoder）和文本编码器（Text Encoder）。
- **一个以图像为基础的编码器**（Image-grounded text encoder）。
- **一个以图像为基础的解码器**（Image-grounded text decoder）。

该模型通过三个损失函数联合进行预训练：

1. **图像-文本对比损失 ITC (Image-Text Contrastive Loss)**：针对图像编码器和文本编码器，通过正负图文对的对比学习，来对齐图像和文本的潜在特征空间。
2. **图像-文本匹配损失 ITM (Image-Text Matching Loss)**：针对以图像为基础的文本编码器，通过对图文匹配性进行二分类，建模图文多模态信息的相关性。
3. **语言建模损失 LM (Language Modeling Loss)**：针对以图像为基础的文本解码器，通过交叉熵损失进行优化，训练模型以自回归的方式生成目标 Caption。


### 1.2 训练方法

网络上获得的大量图文对通常包含许多不准确甚至错误的信息。为了有效利用这种形态的数据，BLIP 提出了 Caption 生成和过滤模块 **CapFilt (Captioning and Filtering)**。它首先从噪声图文对中学习，然后生成和过滤产生新的数据集，再去迭代优化原模型。

CapFilt 包含两个模块，它们都是从预训练的模型初始化，并在人工标注数据集上单独进行微调：

- **Captioner（图像标题生成器）**：基于 *Image-grounded text decoder*。它在人工标注数据集上以 LM 为目标进行微调。对给定的网络图片，Captioner 会生成合成的候选 Caption。
- **Filter（图像标题过滤器）**：基于 *Image-grounded text encoder*。它根据 ITC 和 ITM 的目标进行微调，以学习文本是否与图像匹配，从而去除原始网络文本和合成文本中的噪音文本。


#### Bootstrap 循环过程


```
[网络噪声图文对] ──> [Captioner 生成合成文本]
                        │
                        ▼
[原始文本 + 合成文本] ──> [Filter 噪声过滤] ──> [干净的图文对] + [人工标注数据] ──> [预训练新模型]
```

实验结果表明，通过 Captioner 和 Filter 的协作，BLIP 模型在图像-文本检索、图像标题、视觉问答、视觉推理和视觉对话等各种下游任务上取得了稳定的性能提升。


## 二、 BLIP-2 简介

Salesforce 在 2023 年提出 **BLIP-2**。该模型通过利用**预训练且冻结**的视觉模型和语言模型，在大幅降低训练成本的同时提升了多模态效果。预训练的视觉模型能够提供高质量的视觉表征，而预训练的大语言模型（LLM）则提供了强大的语言生成能力。


### 2.1 模型结构

BLIP-2 由预训练的图像编码器（Image Encoder）、预训练的大语言模型（Large Language Model）和一个可学习的 **Q-Former** 组成。

1. **Image Encoder**：负责从输入图片中提取视觉特征。论文中试验了两种网络结构：基于 CLIP 训练的 `ViT-L/14` 和基于 EVA-CLIP 训练的 `ViT-g/14`。
2. **Large Language Model (LLM)**：负责文本生成。论文中试验了 Decoder-based LLM 和 Encoder-Decoder-based LLM。
3. **Q-Former**：负责**弥合视觉和语言两种模态的差距**。它由 Image Transformer 和 Text Transformer 两个子模块构成，它们**共享相同的自注意力层（Self-Attention）**。
   - **Image Transformer**：通过与图像编码器进行交互提取视觉特征。它的输入是可学习的 **Query（查询向量）**。这些 Query 通过自注意力层相互交互，并通过交叉注意力层（Cross-Attention）与冻结的图像特征交互，还可以通过共享的自注意力层与文本进行交互。
   - **Text Transformer**：作为文本编码器和解码器。它的自注意力层与 Image Transformer 共享。根据具体的预训练任务，应用不同的自注意力掩码（Attention Mask）来控制 Query 和文本的交互方式。



### 2.2 训练方法

为了减少计算成本并避免灾难性遗忘的问题，BLIP-2 在预训练时**完全冻结**了预训练图像模型和语言模型。然而，简单地冻结预训练模型参数会导致视觉特征和文本特征难以对齐。

为此，BLIP-2 提出了**两阶段预训练 Q-Former** 来弥补模态差距：**表示学习阶段**和**生成学习阶段**。


```
阶段一：表示学习 (Vision-to-Language Representation Learning)
┌───────────────┐      ┌────────────┐
│ 冻结 Image Ec │ ──>  │  Q-Former  │  <-- 联合优化 (ITC, ITG, ITM)
└───────────────┘      └────────────┘

阶段二：生成学习 (Vision-to-Language Generative Learning)
┌────────────┐      ┌──────────────┐      ┌───────────────┐
│ Q-Former   │ ──>  │ 线性投影层 FC │ ──>  │ 冻结的大模型LLM │
└────────────┘      └──────────────┘      └───────────────┘
```


### （1）第一阶段：表示学习阶段

在表示学习阶段，将 Q-Former 连接 to 冻结的 Image Encoder，训练集为图像-文本对。通过联合优化三个预训练目标，在 Query 和 Text 之间分别采用不同的注意力掩码策略，从而控制 Image Transformer 和 Text Transformer 的交互方式：

- **① ITC (Image-Text Contrastive Learning - 图像文本对比学习)**
   - **优化目标**：对齐图像嵌入和文本嵌入，将来自 Image Transformer 输出的 Query 嵌入与来自 Text Transformer 输出的文本嵌入对齐。
   - **掩码策略**：为了避免信息泄漏，ITC 采用了**单模态自注意掩码**，不允许 Query 和 Text 相互注意。
   - **匹配逻辑**：Text Transformer 的文本嵌入是 `[CLS]` 标记的输出嵌入，而 Query 嵌入则包含多个输出嵌入。因此，模型首先计算每个 Query 输出嵌入与文本嵌入之间的相似度，然后选择最高的一个作为最终的图像-文本相似度。

- **② ITG (Image-grounded Text Generation - 图像引导文本生成)**
   - **优化目标**：在给定输入图像作为条件的情况下，训练 Q-Former 生成文本，迫使 Query 提取包含文本信息的视觉特征。
   - **结构机制**：由于 Q-Former 的架构不允许冻结的图像编码器和文本标记之间直接交互，因此生成文本所需的信息必须首先由 Query 提取，然后通过自注意力层传递给文本标记。
   - **掩码策略**：ITG 采用**多模态因果注意力（Causal Attention）掩码**来控制交互。Query 可以相互关注，但不能关注 Text 标记；每个 Text 标记都可以处理所有 Query 及其前面的 Text 标记。
   - **特殊标记**：在此任务中，将 `[CLS]` 标记替换为新的 `[DEC]` 标记，作为第一个文本标记来指示解码任务。

- **③ ITM (Image-Text Matching - 图像文本匹配)**
   - **优化目标**：这是一个二元分类任务，通过预测图像-文本对是正匹配（Positive）还是负匹配（Negative），学习图像和文本表示之间的细粒度对齐。
   - **结构机制**：将 Image Transformer 输出的每个 Query 嵌入输入到一个二类线性分类器中以获得对应的 logit，然后将所有的 logit 平均，再计算最终的匹配分数。
   - **掩码策略**：ITM 使用**双向自注意掩码（Bi-directional Attention Mask）**，所有的 Query 和 Text 都可以相互关注。



### （2）第二阶段：生成学习阶段

在生成预训练阶段，将 Q-Former 连接到冻结的 LLM，以充分利用 LLM 强大的语言生成能力。

- **对齐机制**：使用一个全连接层（Fully Connected Layer）将 Q-Former 输出的 Query 嵌入线性投影到与 LLM 的文本嵌入相同的维度，然后将投影后的 Query 嵌入拼接到输入文本嵌入的前面（作为 Soft Prompts）。
- **信息瓶颈作用**：由于 Q-Former 已经在第一阶段经过了预训练，可以提取出包含语言信息的视觉表示，因此它可以有效地充当**信息瓶颈**：只将最有用的视觉特征提供给 LLM，同时过滤掉不相关的视觉信息。这极大地减轻了 LLM 学习视觉-语言对齐的负担。


#### LLM 的实验类型

BLIP-2 测试了两种类型的语言模型：

1. **基于解码器的 LLM (Decoder-based LLM)**：使用**语言建模损失 (Language Modeling Loss)** 进行预训练，其中冻结的 LLM 的任务是根据 Q-Former 提供的视觉表示自回归地生成目标文本。
2. **基于编码器-解码器的 LLM (Encoder-Decoder-based LLM)**：使用**前缀语言建模损失 (Prefix Language Modeling Loss)** 进行预训练。将文本分成两部分：前缀文本（Prefix Text）与视觉表示连接起来作为 LLM 编码器的输入，后缀文本（Suffix Text）则用作 LLM 解码器的生成目标。
