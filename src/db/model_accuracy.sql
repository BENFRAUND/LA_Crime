drop table if exists crime_accuracy_cnt;

CREATE TABLE crime_accuracy_cnt(FBI_Category TEXT PRIMARY KEY, Accurate INT, Inaccurate INT, Total INT);

insert into crime_accuracy_cnt select FBI_Category,
sum(case when fbi_cat_prediction = FBI_Category then 1 else 0 end) as Accurate,
sum (case when fbi_cat_prediction != FBI_Category then 1 else 0 end) as Inaccurate,
count(*)as Total
from new_la_crime_incl_modesc
group by FBI_Category;