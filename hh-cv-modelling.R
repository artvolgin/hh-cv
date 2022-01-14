library(dplyr)
library(lme4)
library(performance)
library(ggplot2)
library(stringr)
library(translateR)
Sys.setlocale(locale = "Russian")
setwd("/data")

# =============================================================================
# 1. PREPROCESSING
# =============================================================================

# Load the data
df_cv <- rio::import("df_cv_skills.xlsx")

# Preprocessing
top_regions <- names(tail(sort(table(df_cv$region_name)), 30))

df_cv <- df_cv %>% 
  filter(expected_salary <= quantile(expected_salary, 0.995, na.rm=T)) %>%
  mutate(year_of_cv_creation = as.factor(year_of_cv_creation))

### Remove applicants from rare universities
freq_uni <- as.data.frame(table(df_cv$university_name))
selected_uni <- as.character((freq_uni %>% filter(Freq >= 30))$Var1)
df_cv <- df_cv %>% filter(university_name %in% selected_uni,
                          !is.na(faculty_category))

# Change the reference categories
df_cv <- df_cv %>%
  mutate(faculty_category = relevel(as.factor(faculty_category),
                                      names(which.max(table(faculty_category)))),
         region_name = relevel(as.factor(region_name),
                                      names(which.max(table(region_name)))),
         professional_area = relevel(as.factor(professional_area),
                                      names(which.max(table(professional_area)))),
         age_group = relevel(as.factor(age_group),
                                      "18–24"),
         expected_salary_category = relevel(as.factor(expected_salary_category),
                                       "nonzero"),
         end_date = relevel(as.factor(as.character(end_date)),
                                        "2000"),
         education_level = relevel(as.factor(education_level),
                                 "unfinished_higher")
         )


# Relevel education level
df_cv$education_level <- factor(df_cv$education_level, levels = c("unfinished_higher", "higher", "bachelor", "master", "candidate"))

# Drop redundant columns
df_cv <- df_cv %>% dplyr::select(-c("...1"))


# Add university-level characteristics
setwd("/data")
df_universities <- rio::import("dataframe_universities.xlsx")
df_universities <- df_universities %>% rename(university_name=name)
df_universities <- df_universities %>% dplyr::select(university_name,
                                                     # mean_ege_score_2019_free,
                                                     specialization_type)

# Translate university names to English
df_universities <- translate(dataset = df_universities,
                     content.field = "university_name",
                     google.api.key = 'GOOGLE_API_KEY',
                     source.lang = 'ru',
                     target.lang = 'en')
df_universities <- df_universities %>% rename(university_name_eng=translatedContent)
df_universities$university_name_eng <- str_replace_all(df_universities$university_name_eng, "&quot;", '"')

# Merge to df_cv
df_cv <- df_cv %>% left_join(df_universities)
df_cv <- df_cv %>% filter(!is.na(specialization_type))

df_cv$specialization_type <- recode(df_cv$specialization_type,
                                   "социально-экономический"="socio-economic",
                                   "классический"="classic",
                                   "педагогический"="pedagogical",
                                   "технический"="technical",
                                   "медицинский"="medical",
                                   "аграрный"="agrarian",
                                   "художественный"="art",
                                   "специализированный"="specialized",
                                   "спортивный"="sport")

df_cv$specialization_type <- relevel(as.factor(df_cv$specialization_type), "classic")

# Save to RDS
setwd(paste0("C:/Users/", Sys.getenv("USERNAME"), "/YandexDisk/WB/jobtrends/data/headhunter/cv"))
saveRDS(df_cv, "df_cv.rds")

# =============================================================================
# 2. MODELLING SKILLS: ICC
# =============================================================================

# Read from RDS
setwd("/data")
df_cv <- readRDS("df_cv.rds")

length(unique(df_cv$university_name))

### ------------------- 2.1 Null Model for each type of skill

skills_icc <- c()
for (y in colnames(df_cv)[startsWith(colnames(df_cv), "skills_")]){
  
  temp_formula <- as.formula(paste0(y, " ~ 1 + (1 | faculty_category / university_name_eng)"))
  
  model_glmer.temp <- glmer(
    temp_formula,
    data = df_cv, family = "binomial", verbose=T, nAGQ=0)
  
  skills_icc <- c(skills_icc, as.numeric(icc(model_glmer.temp)[1]))
  print(y)
  
}

df_skills_icc <- data.frame(name=colnames(df_cv)[startsWith(colnames(df_cv), "skills_")],
                            icc=skills_icc)
saveRDS(df_skills_icc, "df_skills_icc.rds")
df_skills_icc <- readRDS("df_skills_icc.rds")

### PLOT ICC

# Split by soft/hard skills
df_skills_icc$skill_type <- df_skills_icc$name %>% dplyr::recode(
  "skills_social_teamwork"="Soft skills",
  "skills_social_negotiation"="Soft skills",
  "skills_training"="Soft skills",
  "skills_social_presentation"="Soft skills",
  "skills_analytical"="Hard skills",
  "skills_people_managment"="Soft skills",
  "skills_project_managment"="Soft skills",
  "skills_computer_programming"="Hard skills",
  "skills_financial"="Hard skills",
  "skills_medical"="Hard skills",
  "skills_legal"="Hard skills",
  
  "skills_computer_financial_administrative"="Hard skills",
  "skills_smm"="Hard skills",
  "skills_writing"="Hard skills",
  "skills_computer_design_geo"="Hard skills")



# Rename for plotting
df_skills_icc$name <- df_skills_icc$name %>% dplyr::recode(
  "skills_social_teamwork"="Teamwork and Leadership",
  "skills_social_negotiation"="Negotiation",
  "skills_training"="Teaching and Training",
  "skills_social_presentation"="Presentation and Public speaking",
  "skills_computer_specsoftware"="Software",
  "skills_analytical"="Analytical",
  "skills_people_managment"="People managment",
  "skills_project_managment"="Project management",
  "skills_computer_programming"="Programming",
  "skills_financial"="Financial",
  "skills_medical"="Medical-psychological",
  "skills_legal"="Law",
  
  "skills_computer_financial_administrative"="Software for finance, management and analytics",
  "skills_smm"="Social media marketing",
  "skills_writing"="Writing",
  "skills_computer_design_geo"="Software for design, architecture and planning"
  
)



## Barplot with ICC
# Ggplot graph
df_skills_icc %>%
  ggplot(aes(y=reorder(name, icc),
             x=icc,
             fill=factor(skill_type))) +
  geom_bar(stat = "identity") +
  # ggtitle("ICC for different skills") +
  ylab("Skills category") + xlab('Intraclass Correlation Coefficient') +
  scale_fill_manual(name = "skill_type", values=c("coral3","deepskyblue3", "black")) + 
  theme_classic() +
  xlim(0, 0.4) + 
  
  theme(axis.text.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18),
        plot.title = element_text(size=20, face="bold", hjust=0.5),
        legend.position= c(0.8, 0.2),
        legend.title=element_blank(),
        legend.text = element_text(size=14))


# =============================================================================
# 3. COMPUTING MULTILEVEL MODELS
# =============================================================================

skills_list <- colnames(df_cv)[str_starts(colnames(df_cv), "skills_")]

models_list <- list()
i <- 1
for (skill_name in skills_list){
  
  formula.skill <- as.formula(paste0(skill_name,
                            
                            "~ year_of_cv_creation + end_date +
                            log(years_of_experience + 1) + region_name + professional_area +
                            gender + age_group + education_level +
                            
                            faculty_category +
                            
                            specialization_type +
                            
                            (1 | faculty_category / university_name_eng)")
                            
                            )
  
  model_glmer.skill <- glmer(formula.skill,
                             data = df_cv, family = "binomial", verbose=T, nAGQ=0)
  
  models_list[[i]] <- model_glmer.skill
  i <- i + 1
  print(skill_name)
  
}

# Save to RDS
setwd("/data")
saveRDS(models_list, "models_list.rds")


# =============================================================================
# 4. PLOTTING RANDOM INTERCEPTS
# =============================================================================

library(ggh4x)

# Load the models
setwd("/data")
models_list <- readRDS("models_list.rds")

plot.RandomIntercepts <- function(model_object, top_n_coeffs=10, onlySignficant=F){
  
  df_coeffs <- as.data.frame(ranef(model_object, condVar=T))
  
  # Select only university/faculty coefficents
  df_coeffs <- df_coeffs %>% filter(grpvar=="university_name_eng:faculty_category")

  # Replace : with -
  df_coeffs$grp <- str_replace(df_coeffs$grp, ":", " - ")
  
  # Indicate significant coefs
  df_coeffs <- df_coeffs %>%
    dplyr::mutate(condupper=condval + 1.96*condsd,
                  condlower=condval - 1.96*condsd,
                  significant=sign(condlower)==sign(condupper)) # %>%
  
  # Select only signficant coefficents
  if(onlySignficant){
    df_coeffs <- df_coeffs %>% filter(significant)
  }
  
  # Select top N coeffiecents
  df_coeffs <- df_coeffs %>% top_n(top_n_coeffs, wt = condval)
  
  # PLOTTING
  points_color <- ifelse(df_coeffs$significant, "coral3", "darkgrey")
  errorbar_color <- ifelse(df_coeffs$significant, "black", "grey")
  
  ggplot(data = df_coeffs,
         aes(x = condval, y = reorder(grp, condval))) +
    
    geom_errorbarh(aes(xmin = condval - 1.96*condsd,
                       xmax = condval + 1.96*condsd, height = 0.4),
                   color = errorbar_color) +
    geom_point(size = 3, color = points_color) + 
    theme_bw() +
    theme(axis.text.y = element_text(color = "black", size = 16),
          axis.text.x = element_text(size = 16, angle = 50, hjust = 1, face = "bold"),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16))  +
    # geom_vline(xintercept = 0, linetype = "dotted", color = "grey", size = 1.5) + 
    labs(x = "Random Intercept", y = "University - Faculty") + 
    
    # xlim(0,2.5) + 
    
    # TODO: Change the title based on the dependent variable
    # ggtitle("Random Intercepts for Programming skills, TOP-20") +
    theme(plot.title = element_text(size=18, face="bold", hjust=0.5),
          legend.position="bottom") +
    force_panelsizes(rows = unit(4, "in"),
                     cols = unit(5, "in"))
  
}


# 1800
plot.RandomIntercepts(models_list[[12]], onlySignficant=T) # Medical
plot.RandomIntercepts(models_list[[15]], onlySignficant=T) # Software for design, architecture and planning
plot.RandomIntercepts(models_list[[9]], onlySignficant=T) # Programming

plot.RandomIntercepts(models_list[[8]], onlySignficant=T) # Project managment
plot.RandomIntercepts(models_list[[4]], onlySignficant=T) # Presentation


# =============================================================================
# 5. MERGE RANDOM INTERCEPTS IN ONE TABLE
# =============================================================================

# Load the models
setwd("/data")
models_list <- readRDS("models_list.rds")

extract.RandomIntercepts <- function(model_object, top_n_coeffs=10, onlySignficant=F){
  
  df_coeffs <- as.data.frame(ranef(model_object, condVar=T))
  
  # Select only university/faculty coefficents
  df_coeffs <- df_coeffs %>% filter(grpvar=="university_name_eng:faculty_category")
  
  # Replace : with -
  df_coeffs$grp <- str_replace(df_coeffs$grp, ":", " - ")
  
  # Indicate significant coefs
  df_coeffs <- df_coeffs %>%
    dplyr::mutate(condupper=condval + 1.96*condsd,
                  condlower=condval - 1.96*condsd,
                  significant=sign(condlower)==sign(condupper)) # %>%
  
  # Select only signficant coefficents
  if(onlySignficant){
    df_coeffs <- df_coeffs %>% filter(significant)
  }
  
  # Select top N coeffiecents
  df_coeffs <- df_coeffs %>% top_n(top_n_coeffs, wt = condval)
  
  # Add name for the skill variable
  df_coeffs$skill_name <- colnames(model_object@frame)[1]
  
  # Arrange by intercept value
  df_coeffs <- df_coeffs %>% arrange(-condval)
  
  # Select only usefull columns
  df_coeffs <- df_coeffs %>% dplyr::select(grp, condval, condlower, condupper, skill_name)
  
  return(df_coeffs)

}

df_coeffs <- rbind(
  extract.RandomIntercepts(models_list[[12]]),
  extract.RandomIntercepts(models_list[[15]]),
  extract.RandomIntercepts(models_list[[9]]),
  extract.RandomIntercepts(models_list[[8]]),
  extract.RandomIntercepts(models_list[[4]]))
  
rio::export(df_coeffs, "df_coeffs.xlsx")



