
    
    

select
    Ind_id as unique_field,
    count(*) as n_records

from main."stg_applications"
where Ind_id is not null
group by Ind_id
having count(*) > 1


