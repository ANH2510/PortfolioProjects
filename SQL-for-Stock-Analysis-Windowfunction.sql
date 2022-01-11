Select *
From dbo.BankofAmericaDailyStock

--Compare price day by day
With temp_table as (
Select
	Date,
	ROUND(Adj_Close,2) as actual_price,
	ROUND(Lag(Adj_Close,1) OVER (ORDER BY Date),2) as one_day_before_price
From dbo.BankofAmericaDailyStock
)

Select Date, actual_price, one_day_before_price,
		FORMAT((actual_price - one_day_before_price) / one_day_before_price, 'P') as pct_change
From temp_table;


--Compare price by previous month
With temp_table as(
	SELECT year_,month_, avg_month_price,
			ROUND(LAG(avg_month_price,1) OVER (ORDER BY year_,month_),2) as prev_avg_month_price
	FROM
	(
	Select year(Date) as year_, month(Date) as month_,
			ROUND(AVG(Adj_Close),2) as avg_month_price
	From dbo.BankofAmericaDailyStock
	GROUP BY YEAR(Date), MONTH(Date)
	)x
	)

Select year_,month_, avg_month_price, prev_avg_month_price,
		FORMAT((avg_month_price - prev_avg_month_price)/prev_avg_month_price, 'P') as pct_difference
From temp_table
Order by 1,2

