# main.py

import os
import re
import time
import csv
from typing import List, Tuple

from problems.SCQBF.solver.GA_SCQBF import (
    GA_MaxSCQBF_Base,
    GA_MaxSCQBF_LHS,   # EVOL1
    GA_MaxSCQBF_SUS,   # EVOL2
    GA_MaxSCQBF_UX,    # EVOL3
    GA_MaxSCQBF_SS,    # EVOL4
)
from problems.SCQBF.SCQBF import SCQBF

GENERATIONS_DEFAULT = 10000
P1_DEFAULT = 120
P2_DEFAULT = 360
M1_DEFAULT = 0.02
M2_DEFAULT = 0.08

def list_instance_files(dir_: str) -> List[str]:
    if not os.path.isdir(dir_):
        raise FileNotFoundError(f"Diretório de instâncias inválido: {dir_}")
    files = [os.path.join(dir_, f) for f in os.listdir(dir_) if f.lower().endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"Nenhuma instância .txt encontrada em: {dir_}")
    files.sort(key=lambda p: os.path.basename(p))
    return files

def parse_n_k(filename: str) -> Tuple[str, str]:
    m = re.search(r"n_(\d+).*k_(\d+)", filename)
    return (m.group(1), m.group(2)) if m else ("", "")

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

def is_feasible(solution, scqbf: SCQBF) -> bool:
    cover = scqbf.build_cover_count(solution)
    return scqbf.is_feasible_cover(cover)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rodar GA_MaxSCQBF em todas as instâncias e salvar CSV.")
    parser.add_argument("--instances_dir", default=os.path.join(os.getcwd(), "problems", "SCQBF", "instances"))
    parser.add_argument("--out_csv", default=os.path.join("results", "ga_scqbf_results.csv"))

    parser.add_argument("--generations", type=int, default=GENERATIONS_DEFAULT)
    parser.add_argument("--p1", type=int, default=P1_DEFAULT, help="Tamanho de população P1 (PADRÃO).")
    parser.add_argument("--m1", type=float, default=M1_DEFAULT, help="Taxa de mutação M1 (PADRÃO).")
    parser.add_argument("--p2", type=int, default=P2_DEFAULT, help="Tamanho de população P2 (PADRÃO+POP).")
    parser.add_argument("--m2", type=float, default=M2_DEFAULT, help="Taxa de mutação M2 (PADRÃO+MUT).")

    parser.add_argument("--time_limit_s", type=float, default=1800, help="Limite máximo em segundos; interrompe o solve.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    instances_dir = args.instances_dir
    out_csv = args.out_csv
    generations = args.generations
    P1 = args.p1
    M1 = args.m1
    P2 = args.p2
    M2 = args.m2
    time_limit_s = args.time_limit_s
    verbose = args.verbose

    files = list_instance_files(instances_dir)
    ensure_parent_dir(out_csv)

    class Config:
        def __init__(self, name: str, solver_cls, pop_size: int, mut_rate: float):
            self.name = name
            self.solver_cls = solver_cls
            self.pop_size = pop_size
            self.mut_rate = mut_rate

    configs = [
        Config("PADRAO", GA_MaxSCQBF_Base, P1, M1),
        Config("PADRAO+POP", GA_MaxSCQBF_Base, P2, M1),
        Config("PADRAO+MUT", GA_MaxSCQBF_Base, P1, M2),
        Config("PADRAO+EVOL1", GA_MaxSCQBF_LHS,  P1, M1),
        Config("PADRAO+EVOL2", GA_MaxSCQBF_SUS,  P1, M1),
        Config("PADRAO+EVOL3", GA_MaxSCQBF_UX,   P1, M1),
        Config("PADRAO+EVOL4",   GA_MaxSCQBF_SS,  P1, M1),
    ]

    header = [
        "config", "file", "n", "k",
        "generations", "pop_size", "mutation_rate",
        "time_limit_s", "timed_out",
        "max_value", "size", "feasible", "time_s", "elements"
    ]

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        for path in files:
            fname = os.path.basename(path)
            n_str, k_str = parse_n_k(fname)

            for cfg in configs:
                t0 = time.time()

                ga = cfg.solver_cls(
                    generations=generations,
                    pop_size=cfg.pop_size,
                    mutation_rate=cfg.mut_rate,
                    filename=path,
                )
                if hasattr(ga, "verbose"):
                    ga.verbose = verbose

                if time_limit_s is not None:
                    setattr(ga, "time_limit_s", float(time_limit_s))

                best = ga.solve()

                t1 = time.time()
                time_sec = t1 - t0

                timed_out = bool(getattr(ga, "_timed_out", False))
                if time_limit_s is not None and not timed_out:
                    timed_out = (time_sec >= time_limit_s - 1e-9)

                scqbf = SCQBF(path)
                feasible = is_feasible(best, scqbf)
                max_val = float(best.cost)
                elements = " ".join(str(e) for e in list(best))

                writer.writerow([
                    cfg.name, fname, n_str, k_str,
                    generations, cfg.pop_size, f"{cfg.mut_rate:.6f}",
                    f"{time_limit_s:.0f}" if time_limit_s is not None else "",
                    "true" if timed_out else "false",
                    f"{max_val:.6f}", len(best), "true" if feasible else "false",
                    f"{time_sec:.3f}", elements
                ])

                if verbose:
                    print(f"[{cfg.name}] {fname} -> max={max_val:.6f}, size={len(best)}, "
                          f"feasible={feasible}, time={time_sec:.3f}s, timed_out={timed_out}")

    print(f"Resultados (GA) salvos em: {out_csv}")

if __name__ == "__main__":
    main()
