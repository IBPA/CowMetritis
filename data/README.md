## Cow Metritis Cure Risk Study Dataset


1. Filename: *MetritisCureRiskStudy.csv*

2. Number of instances: 550

3. Number of attributes: 24 including the dependent variables

4. Attribute information

```
Dairy: (categorical) Name of the farm.
	- ADC: 0
	- DRU: 1
	- NFH: 2
ID: (numeric) Unique identification number of the animal.
DIMd0: (numeric) Days in milk at the diagnosis of metritis. Day in milk start when a cow has a parturition and initiates a lactation. Metritis generally will occur within the first two weeks after giving birth.
TRT: (categorical) Treated with antibiotic or not.
	- 1: Treated group
	- 0: Non treated group
D0month: (numeric, categorical) Month of the metritis diagnosis.
Season: (categorical) Season of the metritis diagnosis.
	- 0: Cool. Metritis was diagnosed between September and May
	- 1: Hot. Metritis was diagnosed between June and August
Lact: (numeric) How many cycles of lactation cow was having when the information was collected.
Parity: (categorical) Number of birth(s) during the lifetime of the cow. This reflects all previous birth events of the cow including the most recent birth that happened few days before the metritis diagnosis.
	- 1: Primiparous - one birth during the lifetime
	- 2: Multiparous - more than one birth during the lifetime
CalvDif: (numeric, categorical) How hard was the calving event.
	- 1: Unassisted
	- 2: Easy pull / slight problem. One person (can be using chains and handles) but without the use of ropes / pulley.
	- 3: Difficult pull / two people pulling or one person plus moderate use of ropes / pulley
	- 4: Very difficult pull / large amount of force needed. Ropes / pulley essential for delivery of calf
	- 5: C-section or fetotomy
Dyst: (numeric, categorical) Occurrence of dystocia.
	- 0: No dystocia. Unassisted calving. CalvDif = 1.
	- 1: Dystocia. Assisted calving. CalvDif = 2~5.
CalfInfo: (categorical) Gender of the calf.
	- 1: Male
	- 2: Female
	- 3: Twins (i.e. (male, female), (male, male), (female, female))
	- 4: Stillbirth
RFM: (numeric, categorical) Retained fetal membranes. Whether if the cow had failure to expel fetal membranes within 12 hours after parturition.
	- 1: Yes
	- 0: No
BCS5: (numeric) Body condition score at 5 days in milk. Visual evaluation of body fat reserves using a 5-point scale with 0.25-point increments where 1=very thin 5=very fat.
BCS5C: (categorical) Categorical data for body condition score at 5 days in milk.
	- 0: 1 ≤ BCS5 ≤ 3
	- 1: 3 < BCS5 < 3.75
	- 2: 3.75 ≤ BCS5 ≤ 5
VLS: (numeric, categorical) Vaginal laceration due to birth (calving).
	- 0: No
	- 1: Yes and < 2 cm
	- 2: Yes > 2 cm
VLSC: (numeric, categorical) Categorical data for vaginal laceration due to birth (calving).
	- 0: VLS 0 or 1
	- 1: VLS 2
Fevd0: (numeric, categorical) Whether if the animal was or not febrile at the metritis diagnose day.
	- 1: Yes
	- 0: No
Tempd0: (numeric) Rectal temperature (°C) measured on the day of metritis diagnosis.
Milkincrease5DIM: (numeric) Slope of the increase in milk production in the first 05 days of lactation.
Milkincrease7DIM: (numeric) Slope of the increase in milk production in the first 07 days of lactation.
Milkincrease9DIM: (numeric) Slope of the increase in milk production in the first 09 days of lactation.
Daystocure: (numeric) Days from enrolment (metritis diagnosis) to cure.
DIMcure: (numeric) Days in milk for cure. Days from the calving to the day that the cure was diagnosed.
Cured: (numeric, categorical) Whether if the animal cured up to 12 days after enrolment.
	- 1: Cured
	- 0: Not cured
```

5. Missing attribute values

```
CalvDif: 5
Milkincrease5DIM: 87
Milkincrease7DIM: 75
Milkincrease9DIM: 68
Daystocure: 153
DIMcure: 153
```

