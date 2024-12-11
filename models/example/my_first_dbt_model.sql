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
    " + "
    union all
    ".join([f"select {i} as id" for i in __import__('random').sample(range(1, 101), 100)]) + "
)

select *
from source_data

/*
    Uncomment the line below to remove records with null `id` values
*/

where id is not null