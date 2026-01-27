{{
  config(
    materialized='view'
  )
}}

SELECT
    Ind_ID,
    Car_Owner,
    Propert_Owner,
    CHILDREN,
    EDUCATION,
    Annual_income 
    age,
    ABS(DAYS_EMPLOYED) / 365.0 as Employed_years,
    label as is_rejected
FROM {{ source('main', 'raw_credit_data') }}
