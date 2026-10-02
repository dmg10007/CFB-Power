# CFBD advanced box schema

Normalizer: `cfb_power.ingestion.cfbd_advanced.normalize_advanced_box`.

## Assumed input shape

```json
{"teams": {"<category>": [{"team": "Name", "...": "..."}]}}
```

Categories (for example `ppa`, `successRates`, `havoc`) are lists keyed by team name.

## Output

One flat row per team per game. Keys are `game_id`, `team`, then `<category>_<field>` in snake_case, with nested objects joined by underscores (for example `success_rates_overall_total`). Lists are ignored.

## Errors

`AdvancedSchemaError` is raised when `teams` is missing, a category is not a list, or an entry lacks `team`.

## Open items

- The fixture is based on an assumed response shape; replace it with a trimmed real response.
- `CFBD_API_KEY` is the assumed environment variable name.
