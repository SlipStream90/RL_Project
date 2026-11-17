import Link from "next/link";

export default function Home() {
  return (
    <div>
      <h1>Negotiating agents, explained from their own traces.</h1>
      <p className="lede">
        NEXUS runs a five-agent marketplace where reinforcement-learning policies choose
        negotiation actions and every step is recorded. Pick a section to run a live
        episode, inspect an agent, replay a negotiation, or ask why an agent acted.
      </p>
      <div className="panel">
        <h2>Start here</h2>
        <p>Open <Link href="/marketplace">Marketplace</Link> and start an episode.
        Then step through it, replay it, and question it in Explainability.
        Answers cite recorded trajectory values only.</p>
      </div>
      <div className="panel">
        <h2>What is measured</h2>
        <p>Reward decomposition, action distributions, counterfactual rollouts and
        exploit flags all read the same trajectory store. Experiment Lab reports only
        numbers produced by the Week 6 harness.</p>
      </div>
    </div>
  );
}
