/*
    Welcome to your first dbt model!
    Did you know that you can also configure models directly within SQL files?
    This will override configurations stated in dbt_project.yml

    Try changing "table" to "view" below
*/

{{ config(materialized='table') }}

with source_data as (

    select 1 as id
    union all
    {% for i in range(1, 101) %}
    select {{ i }} as id
    {% if not loop.last %} union all {% endif %}
    {% endfor %}
)

select *
from source_data


/*
    Uncomment the line below to remove records with null `id` values
*/

where id is not null