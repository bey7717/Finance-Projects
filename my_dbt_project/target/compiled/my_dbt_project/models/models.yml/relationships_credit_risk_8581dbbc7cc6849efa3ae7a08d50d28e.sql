
    
    

with child as (
    select client_id as from_field
    from main."credit_risk"
    where client_id is not null
),

parent as (
    select Ind_id as to_field
    from main."stg_applications"
)

select
    from_field

from child
left join parent
    on child.from_field = parent.to_field

where parent.to_field is null


