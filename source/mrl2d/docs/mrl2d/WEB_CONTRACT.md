# Web result contract (version 2)

The parent owns production `site/data/results.json` and all final computed assets. The site implementation may use test-only fixtures at an in-memory boundary, never as published CFD data. Initial production JSON may contain the four prescribed cases with `status: "not_run"` and empty result arrays, explicitly pending.

Top level: `schema_version: 2`, `study: "mrl2d"`, `title`, `summary`, `cases` (exactly the four IDs below), `limitations` (strings), `provenance` (public-safe object), `downloads` (label/href). Do not expose internal absolute paths, system usernames or credentials.

IDs in display order: `mrl2d_urans_body`, `mrl2d_urans_permeable`, `mrl2d_les_body`, `mrl2d_les_permeable`.

Each case:
- `id`, `label`, `flow_model` (`SST URANS` or `WALE`), `acoustic_surface` (`on-body` or `permeable`), `status` (`not_run`, `running`, `completed`, `failed`), `summary`.
- `metadata` object with verified values only: `cells`, `ranks`, `requested_end_s`, `achieved_end_s`, `dt_s`, `source_cadence_s`, `density_kg_m3`, `sound_speed_m_s`, `spanwise_model`, `analysis_window_s`, `acoustic_band_hz`, `native_returncode`, `native_End`. Missing values may be null; never replace missing with zero.
- `mesh` object or null: `image`, `caption`.
- `flow` object or null: `video` (optional), `image` (optional poster/GIF), `caption`, `start_s`, `end_s`. Relative URLs only. Use actual generated flow and metadata; a generic geometry animation is not CFD.
- `charts`: at most five semantic figure groups across the page, with `id`, `title`, `description`, `kind` (`line` or `polar`), `x_label`, `y_label`, `traces` (`name`, `x`, `y`, optional `color`, `dash`), `origin: "computed"`, `window`, optional `x_scale`/`y_scale` (`linear` or `log`). Finite numbers or null gaps; matching x/y lengths. Existing `plots.mjs` may be reused if its contract is checked.
- `checks`: `name`, `status`, `detail`.
- `downloads`: `label`, `href`.

Surface comparison and flow-model comparison must use matched support. Label on-body and permeable as alternatives, never summed. Physical pressure of the idealized 2D acoustic field is Pa (not Pa/m); force per unit span is N/m. Sound levels use 1 µPa reference but are not finite-turbine absolute predictions. An acoustic waveform synthesized from selected frequencies is explicitly band-limited/window-conditioned.

The frontend must distinguish missing/failed/computing/completed states, reject malformed identities/unsafe relative asset paths/nonfinite or mismatched arrays, and show no fabricated fallback data. Tests verify actual clicks/keyboard/reduced motion, desktop and 390 px layouts, no console exceptions, and empty/pending handling. Current live public deployment is separate from local site implementation.
