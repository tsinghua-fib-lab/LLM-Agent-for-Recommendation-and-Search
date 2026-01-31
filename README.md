# LLM-Agent-for-Recommendation-and-Search
An index for papers on large language model agents for recommendation and search. 

Please find more details in our survey paper: [A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval](https://arxiv.org/abs/2503.05659).


Please cite our survey paper if you find this index helpful.

```
@article{zhang2025survey,
  title={A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval},
  author={Zhang, Yu and Qiao, Shutong and Zhang, Jiaqi and Lin, Tzu-Heng and Gao, Chen and Li, Yong},
  journal={arXiv preprint arXiv:2503.05659},
  year={2025}
}
```

# Recommendation
![Four domains of LLM Agent's role in recommendation tasks](./figs/Recommend%20Domain.jpg)
| **Domain**    | **Paper**                                 | **What agents can do (ability)**                                                                 |
|---------------|-------------------------------------------|--------------------------------------------------------------------------------------------------|
| Interaction   | RAH! RecSys--Assistant--Human: A Human-Centered Recommendation Framework With LLM Agents [[paper]](https://ieeexplore.ieee.org/abstract/document/10572486/)                  | Assist users in receiving customized recommendations and provide feedback                        |
| Interaction   | Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning   [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657828)            | Use tools for specific recommendation tasks                                                     |
| Interaction   | RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems       [[paper]](https://dl.acm.org/doi/abs/10.1145/3589335.3651242)         | Utilize LLMs as an interface for traditional recommendation tools                                |
| Item          | Agentcf: Collaborative learning with autonomous language agents for recommender systems    [[paper]](https://dl.acm.org/doi/abs/10.1145/3589334.3645537)       | Collaborative learning with user and item agents                                                 |
| Item          | Prospect Personalized Recommendation on Large Language Model-based Agent Platform   [[paper]](https://arxiv.org/abs/2402.18240) | Collaboration between intelligent agent projects and recommender agents                          |
| System        | Recmind: Large language model powered agent for recommendation        [[paper]](https://arxiv.org/abs/2308.14296) | Introduce a self-inspiring algorithm for decision-making                                         |
| System        | Recommender ai agent: Integrating large language models for interactive recommendations    [[paper]](https://arxiv.org/abs/2308.16505)| Integrate LLMs and RSs for interactive recommendations                                           |
| System        | Multi-Agent Collaboration Framework for Recommender Systems        [[paper]](https://arxiv.org/abs/2402.15235)  | Develop a multi-agent collaboration framework for RSs                                            |
| System        | Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning      [[paper]](https://arxiv.org/abs/2403.00843)  | Emphasize long-term user retention using LLM-planned RL algorithms                               |
| System        | A multi-agent conversational recommender system             [[paper]](https://arxiv.org/abs/2402.01135)  | Tackle dialog control and user feedback integration with multi-agent framework                   |
| System        | Lending interaction wings to recommender systems with conversational agents          [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/58cd3b02902d79aea4b3b603fb0d0941-Abstract-Conference.html)     | Combine conversational agents and RSs for better interaction                                     |
| Simulation    | On Generative Agents in Recommendation    [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657844)  | Train LLM agents to simulate real users for evaluation                                           |
| Simulation    | RecAgent: A Novel Simulation Paradigm for Recommender Systems        [[paper]](https://www.researchgate.net/publication/371311704_RecAgent_A_Novel_Simulation_Paradigm_for_Recommender_Systems)  | LLM Agent simulates user behaviors related to the RS                                             |
| Simulation    | Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation    [[paper]](https://arxiv.org/abs/2403.09738) | Use LLMs to simulate users for conversational recommendation tasks                               |
| Simulation    | SUBER: An RL Environment with Simulated Human Behavior for Recommender Systems            [[paper]](https://openreview.net/forum?id=w327zcRpYn)    | Develop an RL environment using LLM to simulate user feedback                                    |
| Simulation    | A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems           [[paper]](https://arxiv.org/abs/2405.08035)        | Propose a framework for LLM-based user simulators in conversational RSs                          |
| Simulation    | Rethinking the evaluation for conversational recommendation in the era of large language models      [[paper]](https://arxiv.org/abs/2305.13112)  | Suggest new evaluation methods using LLMs                                                        |
| Simulation    | How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation      [[paper]](https://dl.acm.org/doi/abs/10.1145/3589335.3651955)   | Examine reliability and limitations of current LLM-based simulators                              |
| Simulation    | Can Large Language Models Be Good Companions? An LLM-Based Eyewear System with Conversational Common Ground      [[paper]](https://dl.acm.org/doi/abs/10.1145/3659600)    | Develop an LLM-based eyewear system with conversational common ground                            |

# Search
![Five domains of LLM Agent's role in search tasks](./figs/Search%20Domain.jpg)
| **Role of agent** | **Paper**                                 | **What agents can do (ability)**                                                                      |
|------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Decomposer       | Knowagent: Knowledge-augmented planning for llm-based agents     [[paper]](https://arxiv.org/abs/2403.03101) |  Integrates knowledge base for task decomposition and logical action execution                          |
| Decomposer       | On the Multi-turn Instruction Following for Conversational Web Agents     [[paper]](https://arxiv.org/abs/2402.15057)    | Utilizes self-reflection memory enhancement planning for web navigation tasks                           |
| Decomposer       | Mindsearch: Mimicking human minds elicits deep ai searcher     [[paper]](https://arxiv.org/abs/2407.20183)    | Learns from experience to decompose tasks and execute multi-step web search                           |
| Decomposer       | Autoact: Automatic agent learning from scratch for qa via self-planning     [[paper]](https://arxiv.org/abs/2401.05268)    | Auto-learns QA agents via self-instruct and trajectory synthesis with multiple sub-agents                           |
| Decomposer       | ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework     [[paper]](https://arxiv.org/abs/2505.18105)    | Modular multi-agent framework for deep web reasoning tasks                           |
| Decomposer       | StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization      [[paper]](https://arxiv.org/abs/2505.15107)      | Step-wise RL optimization for multi-hop QA with dual rewards                               |
| Decomposer       | Kwaiagents: Generalized information-seeking agent system with large language models      [[paper]](https://arxiv.org/abs/2312.04889)      | Hybrid memory bank + toolset for time-sensitive and long-tail information needs                               |
| Decomposer       | Webdancer: Towards autonomous information seeking agency      [[paper]](https://arxiv.org/abs/2505.22648)      | ReAct-style web agent with autonomous multi-step reasoning--action and tool-based web interaction.                               |
| Rewriter    | CoSearchAgent: A Lightweight Collaborative Search Agent with Large Language Models [[paper]](https://arxiv.org/abs/2402.06360)                           | A collaborative agent for query optimization, web search, and cited answer generation.             |
| Rewriter    | TCAF: a multi-agent approach of thought chain for retrieval augmented generation [[paper]](https://openreview.net/forum?id=nvjfWv7uGY)                                                      | A multi-agent RAG framework with reference constraints for complex question answering.             |
| Rewriter    | Doing Personal LAPS: LLM-Augmented Dialogue Construction for Personalized Multi-Session Conversational Search [[paper]](https://arxiv.org/abs/2405.03480)                                     | Uses LLM-guided self-dialogue to collect personalized data with a preference memory.               |
| Rewriter    | Trec ikat 2023: The interactive knowledge assistance track overview [[paper]](https://arxiv.org/abs/2401.01330)                                         | A PTKB-based personalized conversational search pipeline.                                          |
| Rewriter    | Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-Agent LLMs [[paper]](https://dl.acm.org/doi/10.1145/3749099)                                                 | Employs multi-role agents for effective query rewriting and robust ranking.                        |
| Executor    | Llm agents improve semantic code search [[paper]](https://arxiv.org/abs/2408.11058)                                           | An RAG-based agent ensemble for semantic code search.                                              |
| Executor    | Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools [[paper]](https://arxiv.org/abs/2502.04644)                                                 | A multi-agent deep research system integrating mind-map, web-search, and coding agents.            |
| Executor    | PaSa: An LLM Agent for Comprehensive Academic Paper Search [[paper]](https://arxiv.org/abs/2501.10120)                                                      | A dual-agent academic search framework optimized with reinforcement learning.                      |
| Synthesizer | PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents [[paper]](https://arxiv.org/abs/2407.09394)                                                | User-centric agents for personalized and generalizable Retrieval-Augmented Generation (RAG).       |
| Synthesizer | ChatCite: LLM agent with human workflow guidance for comparative literature summary [[paper]](https://arxiv.org/abs/2403.02574)                                                  | Uses reflective generation for comparative literature summaries with structured evaluation.        |
| Synthesizer | Agent-G: An Agentic Framework for Graph Retrieval Augmented Generation [[paper]](https://openreview.net/forum?id=g2C947jjjQ)                                                   | Features a retriever bank and a critic, enabling self-reflection on retrieval-augmented generation.|
| Synthesizer | Weknow-rag: An adaptive approach for retrieval-augmented generation integrating web search and knowledge graphs [[paper]](https://arxiv.org/abs/2408.07611)                                                | Integrates web search and knowledge graphs with multi-stage retrieval and self-assessment.         |
| Simulator   | Usimagent: Large language models for simulating search users [[paper]](https://arxiv.org/abs/2403.09142)                                                 | An LLM-based simulator that models user search behaviors for IR evaluation.                        |
| Simulator   | BASES: Large-scale Web Search User Simulation with Large Language Model based Agents [[paper]](https://arxiv.org/abs/2402.17505)                                                     | Uses user-profiled LLM agents to simulate web search users.                                        |
| Simulator   | ChatShop: Interactive Information Seeking with Language Agents [[paper]](https://arxiv.org/abs/2404.09911)                                                  | Conversational search simulators for evaluation with utterance-level analysis.                     |
| Simulator   | LEMSS: LLM-Based Platform for Multi-Agent Competitive Search Simulation [[paper]](https://dl.acm.org/doi/10.1145/3726302.3730312)                                                     | A multi-agent platform to simulate competitive search environments and analyze dynamics.           |