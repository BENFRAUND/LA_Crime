/*create table fbi_cat
as select crm_cd, crm_cd_desc,
(case when crm_cd < 115 then "Homicide"
	when crm_cd between 121 and 122 then "Rape (121, 122)"
	when crm_cd in (815, 820, 821) then "Rape (815,820,821)"
	when crm_cd between 210 and 220 then "Robbery"
	when crm_cd between 230 and 239 then "Aggravated Assault"
	when crm_cd between 310 and 329 then "Burglary"
	when crm_cd between 510 and 520 then "Motor Vehicle Theft"
	when crm_cd between 330 and 331 then "BTFV"
	when crm_cd between 420 and 421 then "BTFV"
	when crm_cd between 341 and 350 then "Theft"
	when lower(crm_cd_desc) like "%fraud%" or lower(crm_cd_desc) like "%embezzlement%" then "Embezzlement/Fraud"
	when crm_cd >=440 and (lower(crm_cd_desc) like "%theft%" or lower(crm_cd_desc) like "%stolen%" or lower(crm_cd_desc) like "%attempt%") then "Personal/Other Theft"
	when lower(crm_cd_desc) like "%theft%" or lower(crm_cd_desc) like "%stolen%" then "Theft"
	when lower(crm_cd_desc) like "assault" then "Assault"
	else null end) as FBI_Category

from new_la_crime;
*/

/*
create table fbi_cat_tmp2
as select crm_cd, crm_cd_desc, FBI_Category,
(case when FBI_Category in("Homicide","Rape (121, 122)","Rape (815,820,821)","Robbery","Aggravated Assault") then "Violent"
	when FBI_Category in("Burglary","Motor Vehicle Theft","BTFV","Personal/Other Theft") then "Property"
	else "Other" end) as fbi_part_1
from fbi_cat;
*/
drop table la_crime_incl_fbi_cat;

create table la_crime_incl_fbi_cat
as select
dr_no,
area_id,
cast(strftime('%Y%m%d', date_occ) as integer) as date_occ,
cast(strftime('%Y%m%d', date_rptd) as integer) as date_rptd,
longitude,
latitude,
'["'||replace(mocodes,' ','", "')||'"]' as mocodes,
premis_cd,
rpt_dist_no,
status,
hour_occ,
minute_occ,
vict_age,
vict_descent,
vict_sex,
FBI_Category 
from hist_la_crime
left join fbi_cat_tmp2
on hist_la_crime.crm_cd = fbi_cat_tmp2.crm_cd
where fbi_part_1 !="Other"
group by 
dr_no,
area_id,
cast(strftime('%Y%m%d', date_occ) as integer),
cast(strftime('%Y%m%d', date_rptd) as integer),
longitude,
latitude,
'["'||replace(mocodes,' ','", "')||'"]',
premis_cd,
rpt_dist_no,
status,
hour_occ,
minute_occ,
vict_age,
vict_descent,
vict_sex,
FBI_Category
;

select vict_descent from la_crime_incl_fbi_cat group by vict_descent;
