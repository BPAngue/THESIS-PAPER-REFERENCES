# Paper References and Code Prototypes

This repository contains the **reference papers and code prototypes** supporting a thesis on **combinational circuit design using metaheuristic optimization techniques**.

The repository is intended to serve as a centralized collection of the literature used to establish the theoretical foundation of the study, together with the prototype implementations developed as part of the thesis methodology.

## Repository Structure

```text
.
├── Paper_References/
└── codes/
```

### `Paper_References/`

Contains the primary reference papers and related literature reviewed for the thesis.

The references currently focus on three major metaheuristic approaches:

* **Particle Swarm Optimization (PSO)**
* **Simulated Annealing (SA)**
* **Cartesian Genetic Programming (CGP)**

CGP is the **primary metaheuristic approach currently being utilized in the study**. The CGP references provide the main theoretical and methodological foundation for the proposed approach.

The PSO and Simulated Annealing references are included as supporting literature for understanding alternative metaheuristic approaches to optimization and their potential application to combinational circuit design.

### `codes/`

Contains the **code prototypes developed for the thesis methodology**.

These codes are experimental implementations used to explore, test, and validate the proposed methodology. They are developed specifically for the thesis and are not necessarily implementations obtained directly from the referenced papers.

The prototypes may include implementations for:

* Combinational circuit representation
* Circuit evaluation and simulation
* Benchmark file processing
* Fitness evaluation
* Cartesian Genetic Programming
* Metaheuristic optimization procedures
* Mutation and selection mechanisms
* Experimental evaluation of generated circuits

As the thesis progresses, the code prototypes may be modified, extended, or reorganized based on experimental findings.

## Research Focus

The thesis investigates the use of **metaheuristic optimization techniques for combinational circuit design**.

The general goal is to automatically generate combinational circuits that satisfy a given functional specification while potentially optimizing circuit-related objectives such as:

* Functional correctness
* Circuit complexity
* Number of gates or active nodes
* Transistor count
* Propagation delay
* Computational efficiency

## Metaheuristic Approaches

| Method                        | Abbreviation | Role in the Study                                     |
| ----------------------------- | ------------ | ----------------------------------------------------- |
| Particle Swarm Optimization   | PSO          | Primary reference approach                            |
| Simulated Annealing           | SA           | Primary reference approach                            |
| Cartesian Genetic Programming | CGP          | **Primary approach currently utilized in the thesis** |

The repository's CGP literature serves as the main foundation for the current methodology. PSO and Simulated Annealing are maintained as reference approaches for comparison, background study, and potential future experimentation.

## Purpose

This repository is maintained to:

1. Organize the **primary research papers** supporting the thesis.
2. Maintain related literature on metaheuristic approaches to circuit optimization.
3. Develop and test **prototype implementations** of the proposed methodology.
4. Provide a record of the computational work conducted during the research.
5. Support experimentation and future refinement of the proposed combinational circuit design approach.

## Development Status

The repository is **actively under development** as part of the thesis research.

The contents of `codes/` should be considered **research prototypes** and may change substantially as the methodology is refined and experimental results are obtained.

Additional reference papers, algorithms, benchmarks, and experimental implementations will be added as the research progresses.

## Keywords

`Combinational Circuit Design` · `Metaheuristic Optimization` · `Cartesian Genetic Programming` · `CGP` · `Particle Swarm Optimization` · `PSO` · `Simulated Annealing` · `Digital Circuit Optimization` · `Evolutionary Computation`