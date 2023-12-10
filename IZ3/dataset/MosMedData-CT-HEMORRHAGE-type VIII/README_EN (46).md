
<img src="input/logo_en.png" alt="medradiology.moscow" width="35%"/>

CENTER OF DIAGNOSTICS AND TELEMEDICINE

# MosMedData: Expanded set of brain CT scans with and without signs of intracranial hemorrhage, supplemented with clinical and technical parameters

*** 


# About Dataset

> This dataset contains the results of head CT with and without (normal) signs of intracranial hemorrhage (supplemented with clinical and technical parameters). 
The research data were collected in the radiology departments of the Moscow medical
facilities between 05.05.2020 and 08.01.2023.
> - The dataset is enriched with clinical data
> - Signs of ICH
> - Signs of epidural hemorrhage
> - Signs of subarachnoid hemorrhage
> - Signs of subdural hemorrhage
> - Signs of intracerebral hemorrhage
> - Signs of multiple hemorrhages
> - Signs of a skull fracture
> - Signs of combined pathologies
> - Signs of a break in the cerebrospinal fluid spaces
> - Radiology text report
> - The dataset is enriched with and technical data
> - SliceThickness kVp (kiloVolt Peak)
> - XRayTubeCurrent (mA)
> - Convolution Kernel
> - Manufacturer


# Disclaimer

>  Scope of use of this dataset:
>
> - development, training and validation of AI-powered services (incl. those based on the computer vision technologies)
capable of detecting the intracranial Hemorrhage signs;
> - rising awareness among health professionals and the general public.
>
> The license permits **free use (sharing)** of this dataset which includes making copies and sharing on any medium or in any format, **under the following conditions**:
>
> - the dataset must contain a declaration of original authorship which includes:
>   - contributors;
>   - affiliations;
>   - copyright notice;
>   - permanent link to the dataset.
> - link to the license.
>
The license **forbids** the following:
>
> - using this dataset for commercial purposes;
> - sharing a modified or transformed dataset or any new dataset built upon the dataset;
> - providing a fee-based access to the dataset;
> - preventing the sharing of the dataset through any technical methods.

<img src="input/cc.png" alt="medradiology.moscow" width="25%"/>

# General Information

***

## Dataset Title

MosMedData Expanded set of brain CT scans with and without signs of intracranial hemorrhage, supplemented with clinical and technical parameters

## Internal Code

HEMORRHAGE

## Annotation Class

С3

## Verification

Review by a specialist

## Keywords

MosMedData, artificial intelligence, CT, head,  brain, hemorrhage

## Language

English, Russian

## Funding Sources

This dataset was supported by the Russian Science Foundation Grant No. 22-25-20231, https://rscf.ru/project/22-25-20231

## Dataset Version

1.0.0

## Permanent link

[https://mosmed.ai/datasets/](https://mosmed.ai/datasets/)

## Release Date

12.10.2023

# Affiliation and Contributors

## Contributors

> -  Horuzhaja A.N. [1]
> -  Kozlov D.V. [1]
> -  Kremneva E.I. [1]
> -  Novik V.P. [1]
> -  Arzamasov K.M. [1]



## Affiliation

> [1] Research and Practical Clinical Center for Diagnostics and Telemedicine Technologies of the Moscow Health Care Department.

# Data Structure

```
.
|-- dataset_registry.xlsx
|-- README_EN.md
|-- README_RU.md
|-- README_EN.pdf
|-- README_RU.pdf
`-- studies
    |--studyUID_X
   |    |-- seriesUID_X
   |    |    |-- UID_X.dcm
   |    |    |-- UID_X.dcm
   |    |    `-- ...
   |    |
   |    `-- seriesUID_X
   |        |-- UID_X.dcm
   |        |-- UID_X.dcm
   |        `-- ...
   |--studyUID_X
   |    |-- seriesUID_X
   |    |    |-- UID_X.dcm
   |    |    |-- UID_X.dcm
   |    |    `-- ...
   |    |
   |    `-- seriesUID_X
   |        |-- UID_X.dcm
   |        |-- UID_X.dcm
   |        `-- ...
   `--...


```

> - README_EN.md and README_RU.md contain general information about the dataset in the Markdown format in English and Russian languages, respectively. README_EN.pdf and README_RU.pdf contain the same information in the PDF format. 
> - dataset_registry.xlsx contains a list of studies included in the dataset and the path to the corresponding dataset file.
> - The Studies folder contains studyUID-X folders, each of which contains studies in DICOM format.

# Data Overview

| Property      | Value |
| ---------------- | --- |
| Number of studies, pcs.     | 800       |
| Number of patients, ppl.   | 800        |
| Distribution by sex,ppl (M/ F)   | 459/ 338        |
| Distribution by age, years (min./ median/ max.)   | 19/ 57/ 100       |
| Number of studies in each category, psc. (With pathology/ Without pathology)    | 400/ 400   |

# Sharing and Access Information

## License

>  Copyright © 2020 Research and Practical Clinical Center for Diagnostics and Telemedicine Technologies of the Moscow Health Care Department.
> This dataset is licensed under a Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported (CC BY-NC-ND 3.0) License. 
For more information see the LICENSE file or follow this [link](https://creativecommons.org/licenses/by-nc-nd/3.0/) for more information.

## Citation

> Recommended citation:
>
> Datasets of Research and Practical Clinical Center for Diagnostics and Telemedicine Technologies
of the Moscow Health [Electronic resource]. – 2022. – URL: https://mosmed.ai/datasets/
>
> 

## Distribution

> It is prohibited to share this dataset without specifying:
>
> - contributors;
> - affiliations;
> - copyright notice;
> - permanent link to the dataset;
> - link to the license.



