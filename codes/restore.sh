#!/bin/bash

files=(
"refine_job_664907.err"
"connectome_ranking_hexaly_live_improve_igraph_initial_used.xlsx"
"refine_job_664907.out"
"connectome_ranking_hexaly_live_improve_igraph_log 1.txt"
"refine_job_664911.err"
"refine_job_664911.out"
"connectome_ranking_hexaly_live_improve_igraph_log 2.txt"
"refine_job_665817.err"
"refine_job_665817.out"
"connectome_ranking_hexaly_live_log.txt"
"connectome_ranking_hexaly_live.xlsx"
"refine_job_665818.err"
"refine_job_665818.out"
"connectome_ranking_igraph.xlsx"

"connectome_hybrid_phases_direction_changes_log-12cpu.txt"
"connectome_hybrid_phases_direction_changes_log-24cpu 1.txt"
"connectome_hybrid_phases_direction_changes_log-24cpu 2.txt"
"connectome_hybrid_phases_direction_changes_log-24cpu.txt"
"connectome_hybrid_phases_direction_changes_log-48cpu.txt"
"connectome_hybrid_phases_direction_changes_log.txt"

"connectome_hybrid_ranking-12cpu.csv"
"connectome_hybrid_ranking-24cpu 1.csv"
"connectome_hybrid_ranking-24cpu 2.csv"
"connectome_hybrid_ranking-24cpu.csv"
"connectome_hybrid_ranking-48cpu.csv"

"mwfas_511083.out"
"mwfas_connectome.509513.err"
"mwfas_connectome.509513.out"

"output_citeseer_bfsbased_508568.log"
"output_citeseer_bfsbased_509484.log"
"output_mfas_507998.log"
"output_mfas_challenge.ipynb"
"output_minfas_local.ipynb"
"output_pubmed_bfsbased_508569.log"
"output_pubmed_bfsbased_509483.log"
)

trash_files="$HOME/.local/share/Trash/files"
trash_info="$HOME/.local/share/Trash/info"

for f in "${files[@]}"; do
    src="$trash_files/$f"
    info="$trash_info/$f.trashinfo"

    if [ ! -f "$src" ]; then
        echo "⚠️  Missing file: $f"
        continue
    fi
    if [ ! -f "$info" ]; then
        echo "⚠️  Missing metadata: $f"
        continue
    fi

    orig=$(grep "^Path=" "$info" | cut -d= -f2)

    if [ -z "$orig" ]; then
        echo "⚠️  Could not extract original path for: $f"
        continue
    fi

    echo "🔄 Restoring: $f -> $orig"
    mkdir -p "$(dirname "$orig")"
    mv "$src" "$orig"
done

echo "✅ Done restoring files."
