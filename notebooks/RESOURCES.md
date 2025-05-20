# Getting started with molecular simulations as a machine learner  
*May 20, 2025*  

**Disclaimer.** This document contains a curated list of resources I’ve been using to wrap my head around molecular simulations and get excited about sampling algorithms. I’m not a chemist—just an ML person who wandered into this world—so I’ve surely missed things. Still, I hope these pointers help someone like me bridge the gap from code to atoms!  

## Why molecular simulations? A personal account  
When I first heard about MD and Monte Carlo, I thought “Wait, people actually simulate every atom?” It sounded both insane and amazing. I got hooked when I realized these methods let us watch proteins fold, compute binding affinities for drug design, and explore energy landscapes that no human could sketch by hand. It reminded me of playing a physics-based game—so if you like hands-on learning, check out the interactive MD sandbox at [Mol* Viewer](https://molstar.org) and play with a small peptide before reading anything else!  

In my thesis I’ll dive into a fancy twist on Parallel Tempering (T-GePT), but you don’t need that yet—let’s start with the foundations that made me say “Aha!” for the first time.  

---

## 🧪 Statistical Mechanics & Sampling Algorithms  
These give you the “why” behind every move in an MD or MCMC simulation.  

- **Understanding Molecular Simulation (Frenkel & Smit)** — I keep this on my desk; it’s the bible of MC and MD. Whenever detailed balance or ensembles confuse me, I flip to Chapter 2 and everything clicks.  
- **Metropolis et al. (1953)** — The granddaddy of Monte Carlo. Reading this slim paper made me feel connected to the original spark of sampling algorithms.  
- **An Introduction to MCMC for Machine Learning (Andrieu et al., 2003)** — As an ML person, I adored how this survey translates Metropolis–Hastings and Gibbs sampling into our world of probabilistic models.  
- **MIT 2.57 Lecture: Statistical Physics for MD (YouTube)** — A warm, chalk-talk style lecture that explains why we sample with \(e^{-E/kT}\) and what ergodicity really means.  
- **A guide to statistical physics in molecular simulations** — A concise online chapter that refreshed my memory on ensembles and fluctuations before I ever wrote a line of MD code.  

---

## ⚙️ Force Fields & Molecular Mechanics  
Force fields are the “rules of the game” for atoms—get this right, and everything else makes sense.  

- **ParsSilico Blog: “Introduction to Molecular Dynamics Simulations”** — Perfect for a quick, jargon-free overview of bonded vs. nonbonded interactions. I bookmarked it when I needed a crash course on why we care about Lennard-Jones vs. Coulomb terms.  
- **González (2011) “Force fields and molecular dynamics simulations”** — A gentle review of AMBER, CHARMM, and friends. I refer back to its tables of parameters whenever I wonder “How many water models are out there?!”  
- **YouTube: “Intro to Force Fields”** — A 10-minute visual tour through bonds, angles, dihedrals, and van der Waals interactions. The animations really helped me see what’s under the hood.  
- **Behler & Parrinello (2007)** — The paper that taught me neural network potentials *can* learn forces from quantum data. It’s inspiring to see ML stepping in to replace handcrafted formulas!  
- **Computer Simulation of Liquids (Allen & Tildesley)** — A classic textbook that I dip into when I want the gritty details of how energy is computed in practice.  

---

## ⏱️ Integrators & Thermostats  
How do we actually march atoms forward and keep the “temperature” right?  

- **Velocity Verlet Integrator (Wikipedia)** — The workhorse algorithm behind most MD engines. I read the article to understand *why* it conserves energy so well.  
- **OpenMM Docs: Integrators & Langevin Dynamics** — OpenMM’s own guide to choosing an integrator and adding a Langevin thermostat (i.e., random kicks to mimic a heat bath). Super approachable examples in Python!  
- **Nosé–Hoover Thermostat (Nosé 1984 & Hoover 1985)** — I watched a nanoHUB-U lecture on this so the math didn’t scare me. It’s neat to learn how an extra variable can act like a virtual heat reservoir.  
- **Blog: “Cautionary tales of thermostatting in MD”** — A short post I stumbled on that warns how some thermostats (e.g., Berendsen) can warp dynamics if you’re not careful—great to know before you pick one for production runs.  

---

## 🚀 Enhanced Sampling Methods  
When plain MD gets stuck in valleys, these tricks help you jump barriers.  

- **Earl & Deem (2005) “Parallel Tempering”** — The definitive review on replica-exchange. I sketched the algorithm on a whiteboard after reading this and finally *felt* how swapping temperatures helps you escape traps.  
- **CompChem Blog: “Introduction to Enhanced Sampling & Metadynamics”** — A friendly blog with fantastic mountain-and-valley analogies. It’s how I first grasped the idea of “filling up” wells to push your system out.  
- **Laio & Parrinello (2002)** — The original metadynamics paper; short, punchy, and a must-read to see how adding Gaussian hills uncovers free-energy surfaces.  
- **Noé et al. (2019) “Boltzmann Generators”** — A Science paper showing that normalizing flows can directly sample Boltzmann distributions. It blew my mind to see ML generate equilibrium structures without any MD steps!  
- **YouTube: PLUMED Masterclass on Metadynamics** — An hour-long, demo-driven session using alanine dipeptide. Watching someone set up metadynamics in real time was way more instructive than reading slides.  

---

## 📊 Free-Energy Calculations & Analysis  
Free energies are the currency of molecular stability—learn to compute and interpret them.  

- **Mey et al. (2020) “Best Practices for Alchemical Free Energy Calculations”** — A living review and checklist that I used to avoid all the classic pitfalls when I first ran a hydration free-energy calculation.  
- **Zwanzig (1954)** — The one-page introduction to Free Energy Perturbation (FEP). It’s wild that such a concise derivation underpins so many drug-discovery workflows today.  
- **Kumar et al. (1992) “WHAM”** — The Weighted Histogram Analysis Method for combining biased simulations into a smooth free-energy profile. I ran through an online WHAM tutorial with a toy system—highly recommended!  
- **PyEMMA Tutorial on Markov State Models** — A beginner-friendly, code‐centric guide to building MSMs from MD data. It felt like clustering + transition‐matrix estimation, but for your trajectories.  
- **Gilson & Zhou (2007) “Building intuition for binding free energies”** — An opinion piece that helped me understand what free-energy numbers *really* mean—more nuanced than just “lower is better.”  

---

## 🐍 Hands-On Software Tutorials  
Time to get your hands dirty!  

- **OpenMM “Introduction to MD in Python”** — My go-to notebook for setting up a protein in water, running dynamics, and plotting RMSD—all in Python. I still fire this up whenever I need a confidence boost.  
- **GROMACS Tutorial by Justin Lemkul** — The gold standard for command‐line MD. I followed his lysozyme‐in‐water walkthrough step by step and actually got a working simulation on my laptop!  
- **PLUMED Belfast Tutorials** — Exercises (like alanine dipeptide metadynamics) that teach you how to bias simulations and reconstruct free‐energy surfaces with PLUMED hooked into any MD engine.  
- **OpenPathSampling Example Notebooks** — A playground for rare‐event sampling: shooting trajectories, analyzing path ensembles, and experimenting with transition path sampling—all in Python.  

---

## 📦 Datasets & Benchmarks  
Play with real data to train ML models or test sampling methods.  

- **MD17** — Trajectories with quantum energies/forces for small molecules; the canonical benchmark for ML force fields.  
- **rMD17** — A cleaned, consistent update to MD17; preferred by recent ML-potential papers for fair comparisons.  
- **Alanine Dipeptide** — The “hello world” toy system for enhanced sampling: two dihedral angles, two wells, infinite tutorials.  
- **SAMPL Challenges** — Blind host-guest and partition‐coefficient challenges that test the full pipeline: force field choice, sampling, analysis. Great for a reality check!  
- **Anton Trajectories (D. E. Shaw Research)** — Millisecond‐scale protein folding data; perfect for stress‐testing analysis workflows like MSMs or dimensionality reduction.  

---

## 📚 Further Reading & Opinionated Blogs  
A bit of philosophy and community wisdom to round things out.  

- **Westermayr et al. (2021) “Perspective on integrating ML into computational chemistry”** — A lucid survey that convinced me ML really is reshaping force fields, sampling, and analysis.  
- **Prašnikar et al. (2024) “Machine learning heralding a new phase in MD simulations”** — An upbeat but balanced review; I bookmarked their “future directions” section for thesis ideas!  
- **Andrei Klishin’s Blog: “Backdoor to Machine Learning”** — A physicist’s personal take on moving into ML; I loved the analogies and the call to keep one foot in first principles.  
- **Tim Lou (2022) “The Thermodynamics of Machine Learning”** — A playful essay comparing loss landscapes to energy landscapes; a fun way to see how ideas flow between fields.  
- **Community “Hitchhiker’s Guides”** — Keep an eye out on GitHub or Twitter for crowdsourced reading lists—often the fastest way to discover the latest tutorials, notebooks, and opinion pieces.  

---

That’s it! 🎉 If you’re anything like me, you’ll revisit these resources many times over. The molecular simulation community is super welcoming—don’t hesitate to ask questions on forums or Slack channels. Happy exploring, and may your sampling always mix! 🥂  
