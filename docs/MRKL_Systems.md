## MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning

**Authors:** Ehud Karpas, Omri Abend, Yonatan Belinkov, Barak Lenz, Opher Lieber, Nir Ratner, Yoav Shoham, Hofit Bata, Yoav Levine, Kevin Leyton-Brown, Dor Muhlgay, Noam Rozen, Erez Schwartz, Gal Shachaf, Shai Shalev-Shwartz, Amnon Shashua, Moshe Tenenholtz

**Published:** 2022-05-01

**Summary:** Huge language models (LMs) have ushered in a new era for AI, serving as a gateway to natural-language-based knowledge tasks. Although an essential element of modern AI, LMs are also inherently limited in a number of ways. We discuss these limitations and how they can be avoided by adopting a systems approach. Conceptualizing the challenge as one that involves knowledge and reasoning in addition to linguistic processing, we define a flexible architecture with multiple neural models, complemented by discrete knowledge and reasoning modules. We describe this neuro-symbolic architecture, dubbed the Modular Reasoning, Knowledge and Language (MRKL, pronounced "miracle") system, some of the technical challenges in implementing it, and Jurassic-X, AI21 Labs' MRKL system implementation.

## Key Points

1. The Modular Reasoning, Knowledge and Language (MRKL) system is a flexible architecture that combines neural models, such as large language models (LMs), with discrete knowledge and reasoning modules.

2. MRKL systems address the limitations of LMs, such as lack of access to current or proprietary information, lack of reasoning, and model explosion.

3. The architecture consists of an extendable set of modules, called 'experts', and a router that routes natural language input to the best-suited module.

4. These modules can be neural or symbolic, allowing for a more versatile and robust AI system.

5. AI21 Labs has implemented a MRKL system called Jurassic-X, which is being piloted by a few partners.

6. The Jurassic-X system has been trained to extract arguments for basic arithmetic operations with high reliability, demonstrating its potential for production-grade systems.

7. The model's performance has been evaluated on various tasks, including generalization across different number of digits, wordings, question formats, and operations.

8. The results show that Jurassic-X can achieve near-perfect performance in many cases, highlighting its robustness and versatility in handling a wide range of arithmetic problems.