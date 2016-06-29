getRanking <- function(teamname, age_en, joint_en, gender_en, age_es, joint_es, gender_es, gender_nl)
{
  gender = (gender_en + gender_es + gender_nl) / 3
  age = (age_en + age_es) / 2
  joint = (joint_en + joint_es) / 2
  ranking = (gender + age + joint) / 3
  cat(teamname, ranking)
}

getRanking("modaresi16b",0.5128 ,0.3846,0.7564,0.5179,0.4286,0.6964,0.5040)
getRanking("nissim16", 0.5897,0.3846,0.6410,0.5179,0.4286,0.7143, 0.4960)
getRanking("bougiatiotis16", 0.5513,0.3974,0.6923,0.3214,0.2500,0.6786,0.4160)
getRanking("bilan16", 0.4487,0.3333,0.7436,0.4643,0.3750,0.6250, 0.5500)
getRanking("deneva16", 0.3718,0.2051,0.5128,0.3214,0.2679,0.7321,0.6180)


# Output
#modaresi16b 0.5247389
#nissim16 0.5258333
##bougiatiotis16 0.4518944
#bilan16 0.4833944
#deneva16 0.4013556