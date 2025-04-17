*------------------------------------------------------------------------*
*																		 *
*	Passive suicidality in a repressive U.S. political context: Aim II   *
*		aim_ii_lca_event_study.do										 *
*		Simone J. Skeen (04-11-2025)									 *
*																		 *
*------------------------------------------------------------------------*

* housekeeping

clear all
set more off

* install as needed

foreach i in boottest eventdd ftools heatplot matsort palettes colrspace reghdfe schemepack ///
spider xsvmat itsa actest lgraph diff {
	ssc install `i', replace
	}

* set scheme, font

set scheme white_tableau
graph set window fontface "Arial"

* wd

cd "C:\Users\sskee\OneDrive\Documents\02_tulane\01_research\03_dissertation\inputs\data"
clear

* import "short" labeled inference set

import delimited d_inf_labeled_short, varnames(1)

d
*browse

* drop artifacts 
	/// drop pandas implicit idx

drop v1 

* encode subreddits 
	/// p_* prefix not needed here; no post-comments merge in Stata

encode p_sbrt, gen(sbrt) 
drop p_sbrt

*save d_inf_short, replace

save d_inf_labeled_short, replace

///////////////// *-------------------------* /////////////////
///////////////// * Tetrachoric correlation * /////////////////
///////////////// *-------------------------* /////////////////
 
* assign local varlist: indicators `i'
	// adhd aut bpd ptsd - not incl in orig

local i asp dep val sui prg tgd age race dbty  

* tabulate

	*** SJS 2/13: stratified descriptives in aim_i_infer_calibrate.ipynb

tabm `i'

* tetrachorics

tetrachoric `i', stats(rho se)
matrix R = r(Rho)

* heatplot - lower diagonal only

heatplot R, values(format(%9.3f) size(*0.70)) color(hcl diverging, intensity(.6)) ///
	legend(off) aspectratio(1) lower nodiagonal ///
	xlabel(,labsize(*0.60) angle(45)) ylabel(,labsize(*0.60))
	
*/
 
///////////////// *-----------------------------------------* /////////////////
///////////////// * Restrict to pregnancy-capable Redditors * /////////////////
///////////////// *-----------------------------------------* /////////////////
 
keep if prg ==1
d

*use d_model_01, clear

* save

save d_model_01, replace

local i asp dep val sui tgd age race dbty sui 
tabm `i'
tetrachoric `i', stats(rho se)


///////////////// *-----------------------------* /////////////////
///////////////// * Latent class analysis (LCA) * /////////////////
///////////////// *-----------------------------* /////////////////

use d_model_01, clear

* SPSS format for LG

savespss "d_model_01.sav"
 
* assign local varlist: manifest `m'

local m asp dep val sui

* class enumeration: fit indices per _k_ classes (prg==1 restriction, n = 21K) 
 
quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 2) iter(1000) nonrtolerance
estimates store two_class

quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 3) iter(1000) nonrtolerance
estimates store three_class

quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 4) iter(1000) nonrtolerance
estimates store four_class

quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 5) iter(1000) nonrtolerance
estimates store five_class

estimates stats two_class three_class four_class five_class

* _k_ = 3-class solution (optimal human interpretability)
	// k = 4 -> optimal fit stats (incrementally)

quietly gsem (asp dep val sui <- ), family(bernoulli) link(logit) lclass(C 3) iter(1000) ///
	nonrtolerance startvalues(randompr, draws(20) seed(56))
estimates store C3
 
* latent class marginal probabilities 
 
estat lcprob

* goodness of fit

estat lcgof 

* indicator endorsement probabilities

estat lcmean, nose
return list

matrix A = r(table)

*scalar st_mean = A[1,8]
*xsvmat A, rowlabel(rowid) collabel(varname) norestore

* prelim marginsplot

marginsplot, noci

* posterior probability of class membership for each observation

*describe
*drop cpr* predclass maxpr

predict cpr*, classposteriorpr
list asp dep val sui cpr* in 1/30, sep(0)
 
///////////////// *-------------------------* /////////////////
///////////////// * Assign class membership * /////////////////
///////////////// *-------------------------* /////////////////

* gen max posterior probability for each class, assign each ob to predicted class.
		
egen maxpr = rowmax(cpr*)

gen predclass = 1 if cpr1==maxpr
replace predclass = 2 if cpr2==maxpr
replace predclass = 3 if cpr3==maxpr
list asp dep val sui cp* maxpr predclass in 1/30, sep(0)

* class separation

table predclass, statistic(mean cpr1 cpr2 cpr3)

tab predclass
 
* save

save d_model_01, replace 
 
* spider plot 
	/// encodes asp, dep, val, sui as 1, 2, 3, 4, respectively

* display estat lcmean matrix
	
matrix list A	

* manual input for spider

/********* prelim	
clear
input clss strn st_mean
	1 1 .03995494 
	1 2 .0266541
	1 3 .02284164
	1 4 .01454154
	2 1 .41974612
	2 2 .08591563
	2 3 .26095164
	2 4 .06748135
	3 1 .61954479
	3 2 .48244106
	3 3 .12102939
	3 4 .25849802
*/

clear
input clss strn st_mean
	1 1 .0601 
	1 2 .0281
	1 3 .0227
	1 4 .0159
	2 1 .4166
	2 2 .0837
	2 3 .3013
	2 4 .0660
	3 1 .6180
	3 2 .4677
	3 3 .1207
	3 4 .2541

end

label define clssl 1 "Euthymic" 2 "Stagnant-conflicted (mild)" 3 "Stagnant-deprived (moderate)"
label define strnl 1 "asp" 2 "dep" 3 "val" 4 "explicit SI"

label values clss clssl
label values strn strnl

spider st_mean, by(strn) over(clss) alpha(8) msym(none) ra(0(0.1)1) rot(45) smooth(0) sc(black) palette(tol vibrant) lw(0.4)
 
* reload d_model_01 w/ LatentGOLD posterior cl probs

*use d_model_01, clear

clear
import spss using "d_model_01_lg.sav", case(lower) 

*/

///////////////// *-------------------------------* /////////////////
///////////////// * Interrupted time series (ITS) * /////////////////
///////////////// *-------------------------------* /////////////////

* gen class 3 'cl3' outcome 
	// deprecated: now using LG-estimated posterior probs
 
*gen cl3 = (predclass == 3)

* drop LG artifacts

drop _v1
drop cpr1 cpr2 cpr3 predclass

rename _v2 cpr1_lg
rename _v3 cpr2_lg
rename _v4 cpr3_lg
rename _v5 predclass_lg

gen cl3 = (predclass == 3)

*list predclass cl3 in 1/5, sep(0)
*list predclass_lg cl3_lg in 1/5, sep(0)

* inspect datetime vars 
 
list p_date* in 1/5, sep(0)

* gen datevar

gen datevar = date(p_date_str, "MDY", 2099) 
format datevar %td

* gen Stata dates

gen year= year(datevar)
gen month = month(datevar)
gen w = week(datevar)
gen weekly = yw(year,w)
format weekly %tw

* drop artifact

drop p_date_str

* label months

label define monthl 1 "jan" 2 "feb" 3 "mar" 4 "apr" 5 "may" ///
	6 "jun" 7 "jul" 8 "aug" 9 "sep" 10 "oct" 11 "nov" 12 "dec"

label values month monthl

* gen datevar
	/// p_* prefix not needed here; no post-comments merge in Stata

* convert 'p_date' - str to datetime 
	/// double = double-precision numeric fmt
	/// dofc = "date of clock" - converts from milliseconds

gen double datetime = clock(p_date, "YMDhms") 
	
* chronological sort
	/// sort by millisecond datetime var - highest precision

sort datetime

* verify 

list p_date* datetime datevar year month w weekly in 1/5, sep(0)

* count total rows pre-collapse

gen total_rows = 1

* collapse + save: weekly time series (unclustered)

collapse (sum) asp dep val cpnd cl3 tgd age race dbty adhd aut bpd ptsd sui tech total_rows, by(year weekly)
*save d_model_01_its.dta, replace

* collapse + save: weekly time series (panel = prg capability)

		*** SJS 3/17: delete for polished RAP

*collapse (sum) asp dep val cpnd cl3 tgd age race dbty adhd aut bpd ptsd sui tech total_rows, by(prg year weekly)
*save d_model_01_did.dta, replace

* compute pct

foreach v of varlist asp dep val cpnd cl3 tgd age race dbty adhd aut bpd ptsd sui tech {
	gen `v'_pct = (`v' / total_rows) * 100	
	}

* declare weekly time series (unclustered)

tsset weekly, weekly

* tsfill

tsfill
	
* inspect

list year weekly asp asp_pct dep dep_pct val val_pct cpnd cpnd_pct cl3 cl3_pct sui sui_pct if _n <= 20, sep(0)

* save

save d_model_01_its.dta, replace

* export to Python

export delimited using d_model_01_its.csv, replace

///////////////// *------------------------* /////////////////
///////////////// * Auto/cross-correlation * /////////////////
///////////////// *------------------------* /////////////////
 
		*** SJS 2/20: https://www.princeton.edu/~otorres/TS101.pdf for interpretation

* correlogram: autocorrelation 
 
corrgram cpnd_pct, lags(12)
corrgram sui_pct, lags(12)
corrgram cl3_pct, lags(12)

* white noise Q test
	// "white noise" = no autocorrelation
	// H_0: series _no_ serial correlation

wntestq cpnd_pct
wntestq sui_pct
wntestq cl3_pct

		*** SJS 3/7: tl;dr: sui and cl3 have autocorrelation, none have stochastic trends
		*** SJS 2/20: see corellograms for order of autocorrelation

* correlogram: cross-correlation
	// exploratory

*xcorr cpnd_pct sui_pct, lags(10) xlabel(-10(1)10,grid)

* lag selection
	// forecasting-relevant

*varsoc asp_pct sui_pct, maxlag(10)

*line sui_pct weekly

* Augmented Dickey-Fuller test
	// unit root in a series mean = >1 trend in the series
	// H_0: series _has_ unit root

dfuller cpnd_pct, lag(5)
dfuller sui_pct, lag(5)
dfuller cl3_pct, lag(5)

		*** SJS 3/7: p < 0.05 -> stationary
		*** SJS 2/20: _if_ p >0.05 indiciates a unit root, use first-differencing
		
		*** SJS 3/7: stationary throughout

* gen first-differenced outcomes		
		
* n/a
		
*gen asp_pctD1=D1.asp_pct		
*list weekly asp_pct asp_pctD1 in 1/5
		
*dfuller asp_pctD1, lag(5)		


///////////////// *---------------* /////////////////
///////////////// * Model 1a: ITS * /////////////////
///////////////// *---------------* /////////////////

use d_model_01_its.dta, clear
 
* restrict to 2021-01-01 to 2023-12-31 timespan 

drop if year == 2020 | year == 2024
d 
browse

* gen week_n (unclustered)
	/// enumerates all weeks regardless of year

gen week_n = _n
list year weekly week_n asp asp_pct dep dep_pct val val_pct cpnd cpnd_pct cl3 cl3_pct sui sui_pct in 1/5, sep(0)

/*//////////////////////////////////////////////
*
*	T = week_n
*	X_1 = Politico leak (sensitivity analysis)
*	X_2 = Dobbs decision (hypothesized)
*	y_1 = 'cpnd_pct'
*	y_2 = 'sui_pct'
*	y_3 = '<lca_derived_maxpr>'
*
*//////////////////////////////////////////////

* gen treatment indicator: X_1 = Politico leak
	// cf. https://www.epochconverter.com/weeks/2022
	// May 02 2022 -> 2022w18 -> week_n = 70

list year weekly week_n if tin(2022w10,2022w20), sep(0)
		
gen X_1 =.
	replace X_1 = 0 if week_n < 70
	replace X_1 = 1 if week_n >= 70

tab X_1, m

* gen treatment indicator: X_2 = Dobbs decision
	// cf. https://www.epochconverter.com/weeks/2022
	// Jun 24 2022 -> 2022w25 -> week_n = 77
	
		*** SJS 3/7: check viz, set a lag

list year weekly week_n if tin(2022w20,2022w30), sep(0)
	
gen X_2 =.
	replace X_2 = 0 if week_n < 77
	replace X_2 = 1 if week_n >= 77

tab X_2, m

* gen pre-treatment placebo outcomes for Dobbs

gen pl_2 =.
	replace pl_2 = 1 if week_n < 77
	replace pl_2 = 0 if week_n >= 77


**************** Newey-West SEs + compute fitted values

* X_1 = Politico | X_2 = Dobbs (crude)

newey sui_pct c.week_n i.X_2 c.week_n#i.X_2, lag(0)
predict linear_int
actest

* X_1 = Politico | X_2 = Dobbs (I_it coavariate-adjusted)

newey sui_pct c.week_n i.X_2 c.week_n#i.X_2 c.week_n#c.tgd_pct c.week_n#c.age_pct c.week_n#c.race_pct ///
 c.week_n#c.dbty_pct c.week_n#c.adhd_pct c.week_n#c.aut_pct c.week_n#c.bpd_pct c.week_n#c.ptsd_pct, lag(0)
predict linear_int

* Durbin-Watson test
	// H_0 = _no_ serial correlation
	
estat dwatson
estat durbinalt, force

* Breusch-Godfrey test
	// H_0 = _no_ serial correlation

estat bgodfrey	
	
* plot single-group (uncontrolled) ITS

*graph twoway (lfit linear_int week_n if X_2 == 0, lcol(navy)) ///
             (lfit linear_int week_n if X_2 == 1, lcol(navy) ///
             xline(77) ///
             legend(order(1 "pre-Dobbs" 2 "post-Dobbs")))

* est level-change immediately post-treatment

margins, dydx(X_1) at(week_n = 70)

* est slope before and after treatment

margins, dydx(week_n) over(X_1)

* diff in pre- vs post-treatment slopes

margins, dydx(week_n) over(X_1) pwcompare(effects)

drop _cl3_pct _t _x2022w18 _x_t2022w18 _s__cl3_pct_pred

**************** ITSA package

itsa sui_pct, single trperiod(2022w25) lag(0) posttrend figure

		*** SJS 2/18: use help itsa for options

**************** Poisson

poisson cl3 c.week_n i.X_2 c.week_n#i.X_2, vce(robust) irr
newey, lag(0)

*save d_model_01_backup, replace

**************** X_2 -> y_pt2 robustness checking

* inspect

list year weekly week_n X_2 sui sui_pct if tin(2022w20,2022w40), sep(0)

		*** SJS 3/14: face valid; it's right there in the rows

* lag = 1
	// formally tests immediate post-X_2 spike outlier effect

newey sui_pct c.week_n i.X_2 c.week_n#i.X_2, lag(5)
*predict linear_int

itsa sui_pct, single trperiod(2022w25) lag(5) posttrend figure

* drop week_n = 77 (2022w25)
	// formally tests immediate post-X_2 spike outlier effect
	
drop if week_n == 77

list year weekly week_n if tin(2022w20,2022w30), sep(0)
	
gen X_3 =.
	replace X_3 = 0 if week_n < 78
	replace X_3 = 1 if week_n >= 78

newey sui_pct c.week_n i.X_3 c.week_n#i.X_3, lag(0)	

* drop week_n = 78 (2022w26)
	// formally tests immediate post-X_2 spike outlier effect

drop if week_n == 78

list year weekly week_n if tin(2022w20,2022w30), sep(0)
	
gen X_4 =.
	replace X_4 = 0 if week_n < 79
	replace X_4 = 1 if week_n >= 79

newey sui_pct c.week_n i.X_4 c.week_n#i.X_4, lag(0)	

* drop week_n = 79 (2022w27)
	// formally tests immediate post-X_2 spike outlier effect

drop if week_n == 79

list year weekly week_n if tin(2022w20,2022w30), sep(0)
	
gen X_5 =.
	replace X_5 = 0 if week_n < 80
	replace X_5 = 1 if week_n >= 80

newey sui_pct c.week_n i.X_5 c.week_n#i.X_5, lag(0)	

* itsa

itsa sui_pct, single trperiod(2022w26) lag(0) posttrend figure

* Poisson

poisson sui c.week_n i.X_2 c.week_n#i.X_2, vce(robust) irr

* save

save d_model_01_its.dta, replace

* export to Python

export delimited using d_model_01_its.csv, replace

**************** X_2 -> y_pt2 sensitivity analyses

		*** SJS 3/14: TKTK fr PAP

///////////////// *-----------------------------------------* /////////////////
///////////////// * Model 1b: controlled (time-matched) ITS * /////////////////
///////////////// *-----------------------------------------* /////////////////

use d_model_01_its, clear
d

*save d_its_prg, replace

* extract control series: 2021-01-01 to 2021-12-31

keep if year == 2021
browse

* gen Tx group indicator

gen group = 0

*drop linear_int _asp_pct _x2022w25 _x_t2022w25 _s__asp_pct_pred _sui_pct week_n_year

* gen week_n_group: obs week by control group

gen week_n_group = _n

		*** SJS 2/20: in future, use x_1a (model 1a ITS) and x_1b (model 1b CITS) to parsimoniously distinguish

* gen control group X_1
	// cf. https://www.epochconverter.com/weeks/2022
	// May 02 2022 -> 2022w18 -> week_n_group = 18

list year weekly week_n_group if tin(2021w10,2021w20), sep(0)

gen X_1_group =.
	replace X_1_group = 0 if week_n_group < 18
	replace X_1_group = 1 if week_n_group >= 18

tab X_1_group, m

* gen control group X_2
	// cf. https://www.epochconverter.com/weeks/2022
	// Jun 24 2022 -> 2022w25 -> week_n_group = 25

list year weekly week_n_group if tin(2021w20,2021w30), sep(0)
	
gen X_2_group =.
	replace X_2_group = 0 if week_n_group < 25
	replace X_2_group = 1 if week_n_group >= 25

tab X_2_group, m

save d_cits_prg_control, replace

* extract treatment series: 2022-01-01 to 2022-12-31

use d_model_01_its, clear

keep if year == 2022
browse

* gen Tx group indicator

gen group = 1

*drop linear_int _asp_pct _x2022w25 _x_t2022w25 _s__asp_pct_pred _sui_pct week_n_year

* gen week_n_group: obs week by Tx/control group

gen week_n_group = _n

* gen control group X_1
	// cf. https://www.epochconverter.com/weeks/2022
	// May 02 2022 -> 2022w18 -> week_n_group = 18

list year weekly week_n_group if tin(2022w10,2022w20), sep(0)

gen X_1_group =.
	replace X_1_group = 0 if week_n_group < 18
	replace X_1_group = 1 if week_n_group >= 18

tab X_1_group, m

* gen control group X_2
	// cf. https://www.epochconverter.com/weeks/2022
	// Jun 24 2022 -> 2022w25 -> week_n_group = 25

list year weekly week_n_group if tin(2022w20,2022w30), sep(0)
	
gen X_2_group =.
	replace X_2_group = 0 if week_n_group < 25
	replace X_2_group = 1 if week_n_group >= 25

tab X_2_group, m

* save

save d_cits_prg_treatment, replace

* append

clear

append using d_cits_prg_control d_cits_prg_treatment, gen(newv)

d
browse

tsset weekly

* save

save d_model_01_cits, replace

/*//////////////////////////////////////////////
*
*	T = week_n
*	X_1 = Politico leak (sensitivity analysis)
*	X_2 = Dobbs decision (hypothesized)
*	G = group (treatment: 2022; control: 2021)
*	y_1 = 'cpnd_pct'
*	y_2 = 'sui_pct'
*	y_3 = '<lca_derived_maxpr>'
*
*//////////////////////////////////////////////

use d_model_01_cits, clear

**************** Newey-West + compute fitted values

* X_1 = Politico | X_2 = Dobbs

		*** SJS 3/7: NS throughout

newey cl3_pct i.group c.week_n_group i.X_2_group i.group#c.week_n_group i.group#i.X_2_group ///
c.week_n_group#i.X_2_group i.group#c.week_n_group#i.X_2_group, lag(0)

predict linear_int

* X_1 = Politico | X_2 = Dobbs (I_it coavariate-adjusted)

newey cl3_pct i.group c.week_n_group i.X_2_group i.group#c.week_n_group i.group#i.X_2_group ///
c.week_n_group#i.X_2_group i.group#c.week_n_group#i.X_2_group c.week_n#c.tgd_pct c.week_n#c.age_pct ///
c.week_n#c.race_pct c.week_n#c.dbty_pct c.week_n#c.adhd_pct c.week_n#c.aut_pct c.week_n#c.bpd_pct ///
c.week_n#c.ptsd_pct, lag(0)


**************** ITSA package

* tsset panelvar timevar

tsset group week_n_group

itsa sui_pct, treatid(1) trperiod(25) lag(0) posttrend figure

**************** Poisson
	
poisson val i.group c.week_n_group i.X_1_group i.group#c.week_n_group i.group#i.X_1_group ///
c.week_n_group#i.X_1_group i.group#c.week_n_group#i.X_1_group, vce(robust)		
	
	
	
///////////////// *------------------------------------------* /////////////////
///////////////// * Model 1c: diff-in-diff (DiD) event study * /////////////////
///////////////// *------------------------------------------* /////////////////

* reimport d_inf_short
	// Model 1c initializes w/ N = 1.2M analytic sample

use d_inf_labeled_short.dta, clear	
d
browse

		*** SJS 3/14: 99% sure full-sample LCA not needed; delete folded code once confirmed

/******************** reproduce LCA: overall analytic sample

* assign local varlist: manifest `m'

*local m asp dep val sui

* class enumeration: fit indices per _k_ classes (prg==1 restriction, n = 21K) 
 
*quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 2) iter(1000) nonrtolerance
*estimates store two_class

*quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 3) iter(1000) nonrtolerance
*estimates store three_class

*quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(C 4) iter(1000) nonrtolerance
*estimates store four_class

*estimates stats two_class three_class four_class 

* _k_ = 3-class solution (optimal human interpretability)
	// k = 4 -> optimal fit stats (incrementally)

*quietly gsem (asp dep val sui <- ), family(bernoulli) link(logit) lclass(C 3) iter(1000) ///
	nonrtolerance startvalues(randompr, draws(20) seed(56))
*estimates store C3
 
* latent class marginal probabilities 
 
*estat lcprob

* goodness of fit

*estat lcgof 

* indicator endorsement probabilities

*estat lcmean, nose
*return list

*matrix A = r(table)

*scalar st_mean = A[1,8]
*xsvmat A, rowlabel(rowid) collabel(varname) norestore

* prelim marginsplot

*marginsplot, noci

* posterior probability of class membership for each observation

*describe
*drop cpr* predclass maxpr

*predict cpr*, classposteriorpr
*list asp dep val sui cpr* in 1/30, sep(0)
 
 *-------------------------*
 * Assign class membership *
 *-------------------------*

* gen max posterior probability for each class, assign each ob to predicted class.
		
*egen maxpr = rowmax(cpr*)

*gen predclass = 1 if cpr1==maxpr
*replace predclass = 2 if cpr2==maxpr
*replace predclass = 3 if cpr3==maxpr
*list asp dep val sui cp* maxpr predclass in 1/30, sep(0)

* class separation

*table predclass, statistic(mean cpr1 cpr2 cpr3)

*tab predclass
 
save 
 
*/ 
	
* panel format: recreate time series w/ N = 1.2M analytic sample
	 
list p_date* in 1/5, sep(0)

gen datevar = date(p_date_str, "MDY", 2099) 
format datevar %td

gen year= year(datevar)
gen month = month(datevar)
gen w = week(datevar)
gen weekly = yw(year,w)
format weekly %tw

drop p_date_str

label define monthl 1 "jan" 2 "feb" 3 "mar" 4 "apr" 5 "may" ///
	6 "jun" 7 "jul" 8 "aug" 9 "sep" 10 "oct" 11 "nov" 12 "dec"
label values month monthl

gen double datetime = clock(p_date, "YMDhms") 
	
sort datetime

list p_date* datetime datevar year month w weekly in 1/5, sep(0)

* count total rows pre-collapse

gen total_rows = 1

* gen month-year datetime var to avoid "repeated time values within panel" error

gen ym = ym(year, month)
format %tm ym

list ym month year if _n <= 20, sep(0)

* gen exploratory outcome: prg * sui

gen prg_sui = prg * sui
tab prg_sui
browse prg sui prg_sui

* gen exploratory outcome: prg * cpnd

gen prg_cpnd = prg * cpnd
tab prg_cpnd
browse prg cpnd prg_cpnd

* collapse + save: monthly time series (panel = subreddit; T_1 = r/askwomenadvice, r/GirlsSurvivalGuide, r/TwoXChromosomes)

collapse (sum) asp dep val cpnd prg_sui prg_cpnd prg tgd age race dbty adhd aut bpd ptsd sui total_rows, by(sbrt year ym)

* compute pct

foreach v of varlist asp dep val cpnd prg_sui prg_cpnd prg tgd age race dbty adhd aut bpd ptsd sui{
	gen `v'_pct = (`v' / total_rows) * 100	
	}

* restrict to 2021-01-01 to 2023-12-31 timespan 

drop if year == 2020 | year == 2024
*d 
*browse	
		
* declare panel time series (panel = prg capability; for xtdidregress)

xtset sbrt ym

* gen week_n

bysort sbrt: gen ym_n = _n
browse

* inspect

list sbrt ym ym_n year asp asp_pct dep dep_pct val val_pct prg prg_pct cpnd cpnd_pct sui sui_pct if _n <= 20, sep(0)
list sbrt ym ym_n year asp asp_pct dep dep_pct val val_pct prg prg_pct cpnd cpnd_pct sui sui_pct if _n > _N - 20, sep(0)
*xtsum

**************************************

* T_1: _prg_-tailored subreddits: r/askwomenadvice, r/GirlsSurvivalGuide, r/TwoXChromosomes

tab sbrt
tab sbrt, nolabel

gen T_1 = inlist(sbrt, 2, 3, 4)
list sbrt T_1 if _n <= 20, sep(0)
browse

* T_sw: _sui_-tailored subreddit: r/SuicideWatch

gen T_sw = inlist(sbrt, 1)
list sbrt T_sw if _n <= 20, sep(0)
browse

* T_mh: mental health-tailored subreddit: r/depression, r/SuicideWatch

*gen T_mh = inlist(sbrt, 1, 5)
*list sbrt T_mh if _n <= 20, sep(0)
*browse

* D_1: Politico
	// cf. https://www.epochconverter.com/weeks/2022
	// May 02 2022 -> ym = 2022m5 -> ym_n = 17
	
list ym ym_n if tin(2022m1,2022m12), sep(0)	
	
gen post_1 = 0
	replace post_1 = 1 if ym_n >= 17

* D_1 treatment indicator: _prg_-tailored sbrt * post-Politico

gen D_1 = post_1 * T_1	
	
* D_2: Dobbs
	// cf. https://www.epochconverter.com/weeks/2022
	// Jun 24 2022 -> ym = 2022m6 -> ym_n = 18		

list ym ym_n if tin(2022m1,2022m12), sep(0)	
	
gen post_2 = 0
	replace post_2 = 1 if ym_n >= 18

* D_2 treatment indicator: _prg_-tailored sbrt * post-Dobbs

gen D_2 = post_2 * T_1	

* D_3 treatment indicator: r/SW * post-Politico; y = _prg_ | _prg_sui_

gen D_3 = post_1 * T_sw	

* D_4 treatment indicator: r/SW * post-Dobbs; y = _prg_ | _prg_sui_

gen D_4 = post_2 * T_sw	

* D_5 treatment indicator: r/dep, r/SW * post-Politico; y = _prg_ | _prg_sui_

*gen D_5 = post_1 * T_mh	

* D_6 treatment indicator: r//dep, r/SW * post-Dobbs; y = _prg_ | _prg_sui_

*gen D_6 = post_2 * T_mh	

		*** SJS 3/15: these are a bit dredgy; delete

* inspect

browse

* save

save d_model_01_did_xt.dta, replace

* export to Python

export delimited using d_model_01_did_xt.csv, replace	
	
		*** SJS 3/15: 99% sure folded code below is junk
		
		
/* restrict datetime: mimic Model 1a

drop if year == 2020 | year == 2024
browse

* save

save d_did_prg.dta, replace

* gen _post_ indicator: prg==1

keep if prg==1

gen week_n = _n
browse

tsset weekly weekly	

* D_1: Politico
	// cf. https://www.epochconverter.com/weeks/2022
	// May 05 2022 -> 2022w18 -> week_n = 70
	
list year weekly week_n if tin(2022w10,2022w20), sep(0)	
	
gen post_1 = 0
	replace post_1 = 1 if week_n >= 70

* D_2: Dobbs
	// cf. https://www.epochconverter.com/weeks/2022
	// Jun 24 2022 -> 2022w25 -> week_n = 77		

list year weekly week_n if tin(2022w20,2022w30), sep(0)		
	
gen post_2 = 0
	replace post_2 = 1 if week_n >= 77

* save treatment series

save d_did_prg_treatment, replace

* gen _post_ indicator: prg==0	
	
use d_did_prg.dta, clear	
	
* gen _post_ indicator: prg==0

keep if prg==0

gen week_n = _n
browse

tsset weekly weekly	
	
* D_1: Politico
	// cf. https://www.epochconverter.com/weeks/2022
	// May 05 2022 -> 2022w18 -> week_n = 70
	
list year weekly week_n if tin(2022w10,2022w20), sep(0)	
	
gen post_1 = 0
	replace post_1 = 1 if week_n >= 70

* D_2: Dobbs
	// cf. https://www.epochconverter.com/weeks/2022
	// Jun 24 2022 -> 2022w25 -> week_n = 77		

list year weekly week_n if tin(2022w20,2022w30), sep(0)		
	
gen post_2 = 0
	replace post_2 = 1 if week_n >= 77

* save control series

save d_did_prg_control, replace	
	
* append

clear

append using d_did_prg_control d_did_prg_treatment, gen(newv)	
	
* gen treatment indicators	

* D_1: Politico
	
*gen D_1 = .
*	replace D_1 = 1 if weekly >= 2022w18 & prg == 1

gen D_1 = post_1 * prg	
	
* D_2: Dobbs	
	
*gen D_2 = 0
*	replace D_2 = 1 if weekly >= 2022w25 & prg == 1

gen D_2 = post_2 * prg		
	
* save

save d_did_prg.dta, replace	

*/
		
use d_model_01_did_xt.dta, clear

		
**************** OLS (via xtreg: pilot)
	// DID regression (after and treated not needed due to the panel/time fixed effects): https://www.princeton.edu/~otorres/DID101.pdf
	// for reporting, using wildbootstrap xtreg/xtdidregress for ptrends etc

**************** crude	
	
xtreg prg_sui_pct D_4 i.ym, fe vce(cluster sbrt)		

wildbootstrap xtreg prg_sui_pct D_4 i.ym, cluster(sbrt) reps(1000) rseed(56)

**************** adjusted

xtreg prg_sui_pct D_4 i.ym tgd_pct age_pct race_pct dbty_pct adhd_pct aut_pct bpd_pct ptsd_pct, fe vce(cluster sbrt)		

wildbootstrap xtreg prg_sui_pct D_4 i.ym tgd_pct age_pct race_pct dbty_pct adhd_pct aut_pct bpd_pct ptsd_pct, cluster(sbrt) reps(1000) rseed(56)

		
**************** xtdidregress (replicate)		

xtdidregress (prg_sui_pct) (D_4), group(sbrt) time(ym)
*boottest, reps(1000)
		
* viz parallel trends

* xtdidregress native
 
*set scheme s2color 
 
estat trendplots, title("", size(small)) ///
	xtitle("monthly intervals", size(small)) ///
	ytitle("% prg posts expressing explicit SI", size(small)) ///
	xlabel(, angle(45) labsize(tiny)) ///
	legend(order(1 "Control" 2 "Treated") pos(6) rows(1) size(vsmall)) 

*	 plot1opts(lcolor(gray) lpattern(solid) lwidth(medium)) 
*    plot2opts(lcolor(red) lpattern(dash) lwidth(medium)) ///
*    ciopts(lcolor(blue*0.5) lpattern(dash)) ///
*    ci2opts(lcolor(red*0.5) lpattern(dash)) ///
*    note("Data from pre- and post-treatment periods", size(small)) 

* via twoway

*drop mean_sui_pct
bysort ym T_1: egen mean_sui_pct = mean(sui_pct)

twoway line mean_sui_pct ym_n if T_1 == 0, sort || ///
line mean_sui_pct ym_n if T_1 == 1, sort lpattern(dash) ///
legend(label(1 "Control") label(2 "Treated")) ///
xline(17)

* test parallel trends

* xtdidregress native
	// H_0: linear trends are parallel

*estat ptrends
estat ptrends 

* via OLS

reg prg_sui_pct T_sw##ibn.ym if post_2 == 0, vce(cluster sbrt) hascons

* time-to-event var: single event

* : _prg_-tailored sbrt (T_1) * post-Politico (ym_n = 17)

gen t_event_1 = ym_n - 17 if T_1 == 1
	replace t_event_1 = 0 if T_1 == 0

* : _prg_-tailored sbrt (T_1) * post-Dobbs (ym_n = 18)

gen t_event_2 = ym_n - 18 if T_1 == 1		
	replace t_event_2 = 0 if T_1 == 0		
		
browse sbrt ym ym_n t_event_1 t_event_2		
		
* encode time-to-event dummies
	// za = dummy for t_event_1 ("time to event" for Politico)
	// zb = dummy for t_event_2 ("time to event" for Dobbs)

*drop zb1-zb36	
	
tab t_event_2, gen(zb)
		
browse sbrt ym ym_n t_event_2 zb 		
		
sum t_event_2
local min = r(min)
local i = `min'
foreach var of varlist zb1-zb36 {
	label variable `var' "`i'"
	local i = `i'+1
}
		
* event study plot (single event)

*eventdd gdppc, hdfe absorb(country1) vce(cluster country1) timevar(time_to_event)
*graph_op(xlabel(-5(1)6, labsize(3))) ci(rarea, color(gs14%33)) leads(5) lags(6) accum		
		
eventdd cpnd_pct, hdfe absorb(sbrt) vce(cluster sbrt) timevar(t_event_2) ///
graph_op(xlabel(-10(1)10, labsize(3))) ci(rarea, color(gs14%33)) leads(10) lags(10) accum		
		

 *----------------------------------*
 * End of aim_ii_lca_event_study.do *			
 *----------------------------------*



 