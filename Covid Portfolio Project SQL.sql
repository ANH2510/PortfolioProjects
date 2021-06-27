--Covid 19 Data

--Skills used: Joins, CTE's, Temp tables, Window Functions, 
--Aggregate Functions, Creating Views, Converting Data Types

Select *
From PortfolioProject.dbo.CovidDeath
Where continent is not null
order by 3,4

--Select Data to starting with
Select Location, date, total_cases, new_cases, total_deaths, population
From PortfolioProject.dbo.CovidDeath
Where continent is not null
Order by 1,2

--Total Cases vs Total Death
Select Location, date, total_cases, new_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From PortfolioProject.dbo.CovidDeath
Where continent is not null
Order by 1,2

--Total Cases vs Total Deaths in Germany
Select Location, date, total_cases, new_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From PortfolioProject.dbo.CovidDeath
Where location like'Germany'
and continent is not null
Order by 1,2

--Total Cases vs Population
--Infection percentage among countries
Select Location, date, population, total_cases,(total_cases/population)*100 as InfectedPopulationPercentage
From PortfolioProject.dbo.CovidDeath
Order by 1,2

--Countries with Highest Infection rate
Select Location, population, max(total_cases) as HighestInfectionCount, max((total_cases/population)*100) as InfectedPopulationPercentage
From PortfolioProject.dbo.CovidDeath
Group by Location, population
Order by 1,2

--Countries with Highest Death cases
Select Location, max(cast(total_deaths as int)) as TotalDeath
From PortfolioProject.dbo.CovidDeath
Where continent is not null
Group By Location
Order By TotalDeath Desc


--Breaking things down by continent
--Continent with the highest deadth count
Select continent, max(cast(total_deaths as int)) as TotalDeath
From PortfolioProject.dbo.CovidDeath
Where continent is not null
Group By continent
Order By TotalDeath Desc

--Global number
Select SUM(new_cases) as total_cases, sum(cast(total_deaths as int)) as total_deaths
,sum(cast(total_deaths as int))/sum(new_cases)*100 as DeathPercentage
From PortfolioProject.dbo.CovidDeath
Where continent is not null


--Total Population vs Vaccination
--Percentage of population that has been vaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (partition by dea.location order by dea.location
,dea.date) as RollingPeopleVaccinated
From PortfolioProject.dbo.CovidDeath dea
Join PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
order by 2,3

--Using CTE to perform Calculation on Partition By 

with PopvsVac (Continent, Location, Date, Population, New_Vaccination, RollingPeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (partition by dea.location order by dea.location
,dea.date) as RollingPeopleVaccinated
From PortfolioProject.dbo.CovidDeath dea
Join PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
)

Select *, (RollingPeopleVaccinated/Population)*100 as RollingPeopleVaccinatedPercentage
From PopvsVac

--Using Temp table to perform Calculation

Drop Table if exists #VaccinatedPopulationPercentage
Create Table #VaccinatedPopulationPercentage
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_Vaccination numeric,
RollingPeopleVaccinated numeric
)

Insert into #VaccinatedPopulationPercentage
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (partition by dea.location order by dea.location
,dea.date) as RollingPeopleVaccinated
From PortfolioProject.dbo.CovidDeath dea
Join PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null

Select *, (RollingPeopleVaccinated/Population)*100 RollingPeopleVaccinatedPercentage
From #VaccinatedPopulationPercentage

--Creating View to store data for later visulization
Create View VaccinatedPopulationPercentage as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (partition by dea.location order by dea.location
,dea.date) as RollingPeopleVaccinated
From PortfolioProject.dbo.CovidDeath dea
Join PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null

