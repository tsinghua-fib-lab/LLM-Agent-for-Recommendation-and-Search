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
| Interaction   | RAH! RecSys--Assistant--Human: A Human-Centered Recommendation Framework With LLM Agents [[paper]](https://ieeexplore.ieee.org/abstract/document/10572486/)                  | Assists users in receiving customized recommendations and provide feedback                        |
| Interaction   | Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning   [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657828)            | Uses tools for specific recommendation tasks                                                     |
| Interaction   | RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems       [[paper]](https://dl.acm.org/doi/abs/10.1145/3589335.3651242)         | Utilizes LLMs as an interface for traditional recommendation tools                                |
| Representation   | Agentcf: Collaborative learning with autonomous language agents for recommender systems    [[paper]](https://dl.acm.org/doi/abs/10.1145/3589334.3645537)       | Facilitates collaborative learning between user and item agents                                                 |
| Representation   | Prospect Personalized Recommendation on Large Language Model-based Agent Platform   [[paper]](https://arxiv.org/abs/2402.18240) | Controls the collaboration between the Intelligent Agent items and the Agent Recommenders                          |
| Representation   | KGLA: Knowledge Graph Enhanced Language Agents for Recommendation   [[paper]](https://arxiv.org/pdf/2410.19627?) | Improves user agent memory                   |
| System        | Recmind: Large language model powered agent for recommendation        [[paper]](https://arxiv.org/abs/2308.14296) | Introduces a self-inspiring algorithm for decision-making                                         |
| System        | Recommender ai agent: Integrating large language models for interactive recommendations    [[paper]](https://arxiv.org/abs/2308.16505)| Integrates LLMs and RSs for interactive recommendations                                           |
| System        | Multi-Agent Collaboration Framework for Recommender Systems        [[paper]](https://arxiv.org/abs/2402.15235)  | Develops a multi-agent collaboration framework for RSs                                            |
| System        | Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning      [[paper]](https://arxiv.org/abs/2403.00843)  | Emphasizes long-term user retention using LLM-planned RL algorithms                               |
| System        | A multi-agent conversational recommender system             [[paper]](https://arxiv.org/abs/2402.01135)  | Tackles dialog control and user feedback integration with multi-agent framework                   |
| System        | Personalized Recommendation Systems using Multimodal, Autonomous, Multi Agent Systems          [[paper]](https://arxiv.org/pdf/2410.19855)     | Uses multimodal, autonomous, multi-agent systems                                     |
| System        | Lending interaction wings to recommender systems with conversational agents          [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/58cd3b02902d79aea4b3b603fb0d0941-Abstract-Conference.html)     | Combines conversational agents and RSs for better interaction                                     |
| System        | A Hybrid Multi-Agent Conversational Recommender System with LLM and Search Engine in E-commerce          [[paper]](https://dl.acm.org/doi/pdf/10.1145/3640457.3688061)     | Combines LLM agent and search engine to optimize conversational recommendation                   |
| Simulation    | On Generative Agents in Recommendation    [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657844)  | Trains LLM agents to simulate real users for evaluation                                           |
| Simulation    | RecAgent: A Novel Simulation Paradigm for Recommender Systems        [[paper]](https://www.researchgate.net/publication/371311704_RecAgent_A_Novel_Simulation_Paradigm_for_Recommender_Systems)  | Simulates user behaviors related to the RS                                             |
| Simulation    | Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation    [[paper]](https://arxiv.org/abs/2403.09738) | Uses LLMs to simulate users for conversational recommendation tasks                               |
| Simulation    | SUBER: An RL Environment with Simulated Human Behavior for Recommender Systems            [[paper]](https://openreview.net/forum?id=w327zcRpYn)    | Develops an RL environment using LLM to simulate user feedback                                    |
| Simulation    | A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems           [[paper]](https://arxiv.org/abs/2405.08035)        | Proposes a framework for LLM-based user simulators in conversational RSs                          |
| Simulation    | Rethinking the evaluation for conversational recommendation in the era of large language models      [[paper]](https://arxiv.org/abs/2305.13112)  | Suggests new evaluation methods using LLMs                                                        |
| Simulation    | How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation      [[paper]](https://dl.acm.org/doi/abs/10.1145/3589335.3651955)   | Examines reliability and limitations of current LLM-based simulators                              |
| Simulation    | Can Large Language Models Be Good Companions? An LLM-Based Eyewear System with Conversational Common Ground      [[paper]](https://dl.acm.org/doi/abs/10.1145/3659600)    | Develops an LLM-based eyewear system with conversational common ground                            |
| Simulation    | CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent      [[paper]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671837)    | Uses LLM agent to attack LLM-driven RSs                            |
| Simulation    |  LLM-Powered User Simulator for Recommender System     [[paper]](https://arxiv.org/pdf/2412.16984)    | Improves the training efficiency and effectiveness of RSs based on reinforcement learning                        |

# Search
![Five domains of LLM Agent's role in search tasks](./figs/Search%20Domain.jpg)
| **Role of agent** | **Paper**                                 | **What agents can do (ability)**                                                                      |
|------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Decomposer       | Laser: Llm agent with state-space exploration for web navigation       [[paper]](https://arxiv.org/abs/2309.08172)         | Uses state-space exploration for web navigation tasks                                                   |
| Decomposer       | Knowagent: Knowledge-augmented planning for llm-based agents     [[paper]](https://arxiv.org/abs/2403.03101) |  Integrates knowledge base for task decomposition and logical action execution                          |
| Decomposer       | On the Multi-turn Instruction Following for Conversational Web Agents     [[paper]](https://arxiv.org/abs/2402.15057)    | Utilizes self-reflection memory enhancement planning for web navigation tasks                           |
| Decomposer       | A real-world webagent with planning, long context understanding, and program synthesis      [[paper]](https://arxiv.org/abs/2307.12856)      | Learns from experience to complete tasks and divide complex instructions                               |
| Decomposer       | Step: Stacked llm policies for web actions         [[paper]](https://arxiv.org/abs/2310.03720)       | Introduces dynamic strategy combination through task decomposition                                     |
| Decomposer       | Tree Search for Language Model Agents      [[paper]](https://arxiv.org/abs/2407.01476)   | Enhances web navigation using tree search algorithms                                                    |
| Decomposer       | Agent q: Advanced reasoning and learning for autonomous ai agents   [[paper]](https://arxiv.org/abs/2408.07199)  | Integrates MCTS-guided search with self-critique for multi-step reasoning           |
| Rewriter         | CoSearchAgent: A Lightweight Collaborative Search Agent with Large Language Models [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657672)| Enables collaborative search through plug-ins that understand and refine queries                        |
| Rewriter         |Doing Personal LAPS: LLM-Augmented Dialogue Construction for Personalized Multi-Session Conversational Search  [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657815)       | Assists in constructing personalized dialogue datasets to enhance query quality                        |
| Rewriter         | Trec ikat 2023: The interactive knowledge assistance track overview [[paper]](https://arxiv.org/abs/2401.01330)| Utilizes internal knowledge of LLMs for better retrieval and response generation                       |
| Rewriter         | LLM Agents Improve Semantic Code Search [[paper]](https://arxiv.org/abs/2408.11058)| Proposes RAG-powered agents with multi-stream ensemble for semantic code search                       |
| Executor         | AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval    [[paper]](https://arxiv.org/abs/2406.11200)    | Utilizes a comparator LLM to teach the agent how to use tools                                      |
| Executor         | Easytool: Enhancing llm-based agents with concise tool instruction   [[paper]](https://arxiv.org/abs/2401.06201)   | extracts key information from tool documentation and designs a unified interface                                                 |
| Executor         | Executable code actions elicit better llm agents  [[paper]](https://arxiv.org/abs/2402.01030)  | Integrates LLm agents with a Python interpreter in order to execute code actions         |
| Executor         | CodeNav: Beyond tool-use to using real-world codebases with LLM agents     [[paper]](https://arxiv.org/abs/2406.12276)    | Proposes code-as-tool paradigm through semantic code search engines                                 |
| Synthesizer      | PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents [[paper]](https://arxiv.org/abs/2407.09394) | Uses real-time personalized data to enhance the relevance of the returned results  |
| Synthesizer      | ChatCite: LLM agent with human workflow guidance for comparative literature summary [[paper]](https://arxiv.org/abs/2403.02574) | Mimics human methods to extract key points and write summaries for literature reviews
| Synthesizer      | PaSa: An LLM Agent for Comprehensive Academic Paper Search [[paper]](https://arxiv.org/abs/2501.10120) | Utilizes a selector to determine whether search results should be included or not                           |
| Simulator        | Analysing utterances in llm-based user simulation for conversational search [[paper]](https://dl.acm.org/doi/10.1145/3650041) | Explores user emulators in conversation search systems for multi-round clarification                    |
| Simulator        | Usimagent: Large language models for simulating search users  [[paper]](https://arxiv.org/abs/2403.09142)   | Simulates users' query, click, and stop behavior in search tasks                                          |
| Simulator        | BASES: Large-scale Web Search User Simulation with Large Language Model based Agents  [[paper]](https://arxiv.org/abs/2402.17505)   | Establishes a parameterized user profiling system with validation framework                                           |
| Simulator        | ChatShop: Interactive Information Seeking with Language Agents  [[paper]](https://arxiv.org/abs/2404.09911)   | Introduces LLM-simulated shoppers to evaluate agents' multi-turn interaction                                          |
