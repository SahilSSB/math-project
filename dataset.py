import networkx as nx
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
import os

# CONFIG
N_EMPLOYEES = 500          # org size
BA_M = 3                   # BA attachment (higher = more hub-heavy)
ER_NOISE_P = 0.002         # background random noise edge probability
N_CLIQUES = 1              # no. of separate insider rings to plant
CLIQUE_SIZE = 6            # nodes per planted clique
CLIQUE_DENSITY = 0.85      # edge probability WITHIN the planted clique (1.0 = perfect clique)
SIM_DAYS = 60               # number of days of activity to simulate
NORMAL_EDGE_WEIGHT_RANGE = (500, 50_000)      # bytes per normal email/attachment
INSIDER_EDGE_WEIGHT_RANGE = (200, 3_000)      # insiders send SMALL amounts (that's the point)
INSIDER_EDGE_FREQ_MULTIPLIER = 4               # insiders communicate more often with each other
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def buildBackgroundGraph(n, m, noiseP):
    g = nx.barabasi_albert_graph(n, m, seed=RANDOM_SEED)
    noise = nx.erdos_renyi_graph(n, noiseP, seed=RANDOM_SEED)
    g.add_edges_from(noise.edges())
    return g


def plantCliques(g, nCliques, cliqueSize, density):
    n = g.number_of_nodes()
    allNodes = list(g.nodes())
    insiderSets = []

    for _ in range(nCliques):
        available = [x for x in allNodes if not any(x in s for s in insiderSets)]
        cliqueNodes = random.sample(available, cliqueSize)
        insiderSets.append(set(cliqueNodes))

        for i in range(len(cliqueNodes)):
            for j in range(i + 1, len(cliqueNodes)):
                if random.random() < density:
                    g.add_edge(cliqueNodes[i], cliqueNodes[j])

    return g, insiderSets


def generateEventLog(g, insiderSets, simDays):
    allInsiders = set().union(*insiderSets) if insiderSets else set()
    startDate = datetime(2026, 1, 1)
    events = []

    for u, v in g.edges():
        isInsiderEdge = (u in allInsiders) and (v in allInsiders) and \
                           any(u in s and v in s for s in insiderSets)

        if isInsiderEdge:
            nEvents = np.random.poisson(lam=8 * INSIDER_EDGE_FREQ_MULTIPLIER)
            weightRange = INSIDER_EDGE_WEIGHT_RANGE
        else:
            nEvents = np.random.poisson(lam=3)
            weightRange = NORMAL_EDGE_WEIGHT_RANGE

        for _ in range(max(nEvents, 0)):
            dayOffset = random.uniform(0, simDays)
            ts = startDate + timedelta(days=dayOffset,
                                         hours=random.uniform(0, 24))
            sizeBytes = random.randint(*weightRange)

            sender, recipient = (u, v) if random.random() < 0.5 else (v, u)

            events.append({
                "senderId": f"EMP_{sender:04d}",
                "recipientId": f"EMP_{recipient:04d}",
                "timestamp": ts.isoformat(),
                "bytes": sizeBytes,
                "isInsiderEvent": isInsiderEdge,
            })

    df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    return df, allInsiders


def main():
    g = buildBackgroundGraph(N_EMPLOYEES, BA_M, ER_NOISE_P)
    g, insiderSets = plantCliques(g, N_CLIQUES, CLIQUE_SIZE, CLIQUE_DENSITY)
    df, allInsiders = generateEventLog(g, insiderSets, SIM_DAYS)

    outputDir = "outputs"
    os.makedirs(outputDir, exist_ok=True)

    detectorInput = df.drop(columns=["isInsiderEvent"])
    detectorInput.to_csv(os.path.join(outputDir, "syntheticOrgLog.csv"), index=False)

    labels = pd.DataFrame({
        "nodeId": [f"EMP_{i:04d}" for i in range(N_EMPLOYEES)],
    })
    labels["isInsider"] = labels["nodeId"].apply(
        lambda x: int(int(x.split("_")[1]) in allInsiders)
    )
    cliqueMap = {}
    for idx, s in enumerate(insiderSets):
        for node in s:
            cliqueMap[f"EMP_{node:04d}"] = idx
    labels["cliqueId"] = labels["nodeId"].map(cliqueMap).fillna(-1).astype(int)
    labels.to_csv(os.path.join(outputDir, "groundTruthLabels.csv"), index=False)

    meta = {
        "nEmployees": N_EMPLOYEES,
        "baM": BA_M,
        "erNoiseP": ER_NOISE_P,
        "nCliques": N_CLIQUES,
        "cliqueSize": CLIQUE_SIZE,
        "cliqueDensity": CLIQUE_DENSITY,
        "simDays": SIM_DAYS,
        "randomSeed": RANDOM_SEED,
        "nEventsTotal": len(df),
        "nInsiderEvents": int(df["isInsiderEvent"].sum()),
        "insiderNodeIds": sorted([f"EMP_{n:04d}" for n in allInsiders]),
    }
    with open(os.path.join(outputDir, "datasetMetadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated {len(df)} events across {N_EMPLOYEES} employees.")
    print(f"Planted {N_CLIQUES} clique(s) of size {CLIQUE_SIZE}, "
          f"density {CLIQUE_DENSITY} -> insiders: {sorted(allInsiders)}")
    print(f"Insider events: {meta['nInsiderEvents']} "
          f"({100*meta['nInsiderEvents']/len(df):.2f}% of total traffic)")


if __name__ == "__main__":
    main()
