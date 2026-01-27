-- income per child
-- annual income / # of children

WITH source AS (
    SELECT * FROM main."raw_credit_data"
)

SELECT Ind_ID as client_id, Annual_income, CHILDREN,
CASE 
    WHEN children > 0 THEN Annual_income / CHILDREN
    ELSE NULL
END AS income_per_child
FROM source