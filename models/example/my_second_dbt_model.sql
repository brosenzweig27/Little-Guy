
-- Use the `ref` function to select from other models

with mods as (
    select id
         , MOD(id, 3) as mod3
         , MOD(id, 7) as mod7
      from {{ ref('my_first_dbt_model') }}
)
select *
from mods
where mod3 = 0
order by id, mod7
