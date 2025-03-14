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
| Decomposer       | Laser: Llm agent with state-space exploration for web navigation       [[paper]](https://arxiv.org/abs/2309.08172)         | Use state-space exploration for web navigation tasks                                                  |
| Decomposer       | Knowagent: Knowledge-augmented planning for llm-based agents     [[paper]](https://arxiv.org/abs/2403.03101) | Integrate knowledge base for task decomposition and logical action execution                          |
| Decomposer       | On the Multi-turn Instruction Following for Conversational Web Agents     [[paper]](https://arxiv.org/abs/2402.15057)    | Utilize self-reflection memory enhancement planning for web navigation tasks                          |
| Decomposer       | A real-world webagent with planning, long context understanding, and program synthesis      [[paper]](https://arxiv.org/abs/2307.12856)      | Learn from experience to complete tasks and divide complex instructions                               |
| Decomposer       | Heap: Hierarchical policies for web actions using llms         [[paper]](https://arxiv.org/abs/2310.03720)       | Introduce dynamic strategy combination through task decomposition                                     |
| Decomposer       | Tree Search for Language Model Agents      [[paper]](https://arxiv.org/abs/2407.01476)   | Enhance web navigation using tree search algorithms                                                   |
| Decomposer       | React: Synergizing reasoning and acting in language models        [[paper]](https://arxiv.org/abs/2210.03629)  | Overcome illusions and error propagation in chain reasoning with simple Wiki API interaction           |
| Rewriter         | CoSearchAgent: A Lightweight Collaborative Search Agent with Large Language Models [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657672)| Enable collaborative search through plug-ins that understand and refine queries                        |
| Rewriter         |Doing Personal LAPS: LLM-Augmented Dialogue Construction for Personalized Multi-Session Conversational Search  [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657815)       | Assist in constructing personalized dialogue datasets to enhance query quality                        |
| Rewriter         | Trec ikat 2023: The interactive knowledge assistance track overview [[paper]](https://arxiv.org/abs/2401.01330)| Utilize internal knowledge of LLMs for better retrieval and response generation                       |
| Executor         | AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval    [[paper]](https://arxiv.org/abs/2406.11200)    | Present a tool-assisted framework for precise knowledge retrieval                                      |
| Executor         | Openagents: An open platform for language agents in the wild    [[paper]](https://arxiv.org/abs/2310.10634)   | Incorporate over 200 daily API for diverse tasks                                                 |
| Executor         | ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval   [[paper]](https://arxiv.org/abs/2403.06551)  | A Self-Adaptation and hierarchical awareness rearrangement method for improving tool retrieval         |
| Executor         | Walert: Putting Conversational Information Seeking Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot     [[paper]](https://dl.acm.org/doi/abs/10.1145/3627508.3638309)    | Retrieving Information from the Knowledge Base to get better results                                   |
| Synthesizer      | Know where to go: Make LLM a relevant, responsible, and trustworthy searcher     [[paper]](https://arxiv.org/abs/2310.12443)       | Propose a generative retrieval framework to promote query-source connection                            |
| Synthesizer      | WILBUR: Adaptive In-Context Learning for Robust and Accurate Web Agents   [[paper]](https://arxiv.org/abs/2404.05902)    | Utilize steps to learn from task examples and perform intelligent backtracking                        |
| Synthesizer      | PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents [[paper]](https://arxiv.org/abs/2407.09394) | Continuously refine understanding of user requests with real-time user data                           |
| Simulator        | Analysing utterances in llm-based user simulation for conversational search [[paper]](https://dl.acm.org/doi/abs/10.1145/3650041) | Explore user emulators in conversation search systems for multi-round clarification                    |
| Simulator        | Usimagent: Large language models for simulating search users  [[paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3657963)   | Simulate user query, click, and stop behavior in search tasks                                          |
