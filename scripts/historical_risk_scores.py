"""Historical monthly Risk Score reconstruction 1994-2026.

Each row is a month-1 assessment using only information that was visible
at that month start or already visible from prior months. Avoid lookahead:
late-month crashes, policy announcements, bottoms, and rescue results are
not treated as known at the first day of that same month.

Scores follow .claude/skills/crash-survival-assessment/SKILL.md:
  - Duration axis: 0=Fast, 25=Medium, 50=Slow
  - Breadth axis: 0=Contained, 25=Mixed, 50=Universal
  - Risk Score = Duration + Breadth, therefore {0, 25, 50, 75, 100}
  - Recommendation: continue for 0/25, reduce for 50, stop for 75/100

The gate rule is critical: duration stays Fast unless banks are visibly
the patient, a funding/mark-to-market cliff is already visible, or Fed
paralysis is actually being tested. Inflation, low rate space, or bubble
valuation alone do not trigger Slow duration.

To regenerate CSV and the latest monthly report:
python3 scripts/historical_risk_scores.py
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

PRODUCTION_STOP_THRESHOLD = 75

# Schema: (year_month, duration_score, breadth_score, key_event, confidence)
HISTORICAL_RISK_ROWS: list[tuple[str, int, int, str, str]] = [
    ('1994-01', 0, 0, 'Pre-tightening calm; Fed hike not yet known', 'H'),
    ('1994-02', 0, 0, 'Fed tightening risk visible; first hike not yet known', 'H'),
    ('1994-03', 0, 25, '1994 bond rout visible after first Fed hike', 'H'),
    ('1994-04', 0, 25, 'Bond rout and Fed-tightening shock continuing', 'H'),
    ('1994-05', 0, 25, 'Bond rout continuing; banks healthy', 'H'),
    ('1994-06', 0, 25, 'Fed tightening and bond-volatility stress continuing', 'H'),
    ('1994-07', 0, 25, 'Bond-market stress and dollar weakness visible', 'H'),
    ('1994-08', 0, 25, 'Fed tightening cycle still repricing bonds', 'H'),
    ('1994-09', 0, 25, 'Bond rout still the dominant macro stress', 'H'),
    ('1994-10', 0, 25, 'Fed tightening and bond losses still visible', 'H'),
    ('1994-11', 0, 25, 'Large Fed hike expected around tightening cycle', 'H'),
    ('1994-12', 0, 25, 'Bond rout and Mexico-peso fragility visible; late-month devaluation not yet known', 'H'),
    ('1995-01', 0, 25, 'Mexico peso crisis visible from prior month', 'H'),
    ('1995-02', 0, 25, 'Mexico support package / EM aftershock still visible', 'H'),
    ('1995-03', 0, 0, 'Soft-landing recovery; stress contained', 'H'),
    ('1995-04', 0, 0, 'Calm soft-landing phase', 'H'),
    ('1995-05', 0, 0, 'Calm bull recovery', 'H'),
    ('1995-06', 0, 0, 'Calm bull recovery', 'H'),
    ('1995-07', 0, 0, 'Fed easing expectation; broad stress absent', 'H'),
    ('1995-08', 0, 0, 'Calm bull recovery', 'H'),
    ('1995-09', 0, 0, 'Calm bull recovery', 'H'),
    ('1995-10', 0, 0, 'Calm bull recovery', 'H'),
    ('1995-11', 0, 0, 'Calm bull recovery', 'H'),
    ('1995-12', 0, 0, 'Calm year-end', 'H'),
    ('1996-01', 0, 25, 'Mexico/EM aftershock risk visible; US banks healthy', 'H'),
    ('1996-02', 0, 25, 'EM aftershock risk fading but still visible', 'H'),
    ('1996-03', 0, 0, 'Calm bull market', 'H'),
    ('1996-04', 0, 0, 'Calm bull market', 'H'),
    ('1996-05', 0, 0, 'Calm bull market', 'H'),
    ('1996-06', 0, 0, 'Calm bull market', 'H'),
    ('1996-07', 0, 0, 'Calm bull market', 'H'),
    ('1996-08', 0, 0, 'Calm bull market', 'H'),
    ('1996-09', 0, 0, 'Calm bull market', 'H'),
    ('1996-10', 0, 0, 'Calm bull market', 'H'),
    ('1996-11', 0, 0, 'Calm bull market', 'H'),
    ('1996-12', 0, 0, 'Calm year-end', 'H'),
    ('1997-01', 0, 0, 'Calm bull market', 'H'),
    ('1997-02', 0, 0, 'Calm bull market', 'H'),
    ('1997-03', 0, 0, 'Calm bull market', 'H'),
    ('1997-04', 0, 0, 'Calm bull market', 'H'),
    ('1997-05', 0, 0, 'Calm bull market', 'H'),
    ('1997-06', 0, 0, 'Asian currency fragility visible but not yet broad', 'M'),
    ('1997-07', 0, 0, 'Asia currency fragility visible; Thailand break not yet known', 'H'),
    ('1997-08', 0, 25, 'Asian financial crisis visible; US banks healthy', 'H'),
    ('1997-09', 0, 25, 'Asian crisis and EM risk-off continuing', 'H'),
    ('1997-10', 0, 25, 'Asian crisis visible; late-month equity break not yet known', 'H'),
    ('1997-11', 0, 25, 'Asian crisis contagion visible', 'H'),
    ('1997-12', 0, 25, 'Asian crisis still pressuring EM', 'H'),
    ('1998-01', 0, 25, 'Asian crisis tail still visible', 'H'),
    ('1998-02', 0, 25, 'Asian crisis tail still visible', 'H'),
    ('1998-03', 0, 25, 'Asian crisis tail fading', 'H'),
    ('1998-04', 0, 0, 'Calm recovery from Asian crisis tail', 'H'),
    ('1998-05', 0, 0, 'Calm bull market', 'H'),
    ('1998-06', 0, 25, 'Russia/EM funding stress becoming visible', 'H'),
    ('1998-07', 0, 25, 'Russia/EM stress visible', 'H'),
    ('1998-08', 0, 25, 'Russia stress visible; default not yet known', 'H'),
    ('1998-09', 25, 25, 'Post-Russia default funding stress; LTCM risk visible', 'H'),
    ('1998-10', 25, 25, 'LTCM rescue/Fed-cut stress window still visible', 'H'),
    ('1998-11', 0, 25, 'Post-LTCM calming but EM stress still visible', 'H'),
    ('1998-12', 0, 0, 'Calm year-end after Fed response', 'H'),
    ('1999-01', 0, 0, 'Post-LTCM calm restored', 'H'),
    ('1999-02', 0, 0, 'Calm dot-com bull market', 'H'),
    ('1999-03', 0, 0, 'Calm dot-com bull market', 'H'),
    ('1999-04', 0, 0, 'Calm dot-com bull market', 'H'),
    ('1999-05', 0, 0, 'Calm dot-com bull market', 'H'),
    ('1999-06', 0, 0, 'Fed tightening risk but no crash catalyst', 'H'),
    ('1999-07', 0, 0, 'Calm dot-com bull market', 'H'),
    ('1999-08', 0, 0, 'Calm dot-com bull market', 'H'),
    ('1999-09', 0, 0, 'Y2K liquidity planning visible; Fed has response room', 'H'),
    ('1999-10', 0, 0, 'Y2K liquidity planning visible; broad stress absent', 'H'),
    ('1999-11', 0, 0, 'Y2K liquidity planning visible; broad stress absent', 'H'),
    ('1999-12', 0, 25, 'Y2K liquidity risk visible but Fed backstop expected', 'H'),
    ('2000-01', 0, 0, 'Dot-com melt-up; crash catalyst not yet visible', 'H'),
    ('2000-02', 0, 0, 'Dot-com melt-up; banks healthy', 'H'),
    ('2000-03', 0, 25, 'Dot-com valuation and Fed-tightening fragility visible', 'H'),
    ('2000-04', 0, 25, 'Tech unwind visible after dot-com peak', 'H'),
    ('2000-05', 0, 25, 'Tech unwind and Fed tightening visible', 'H'),
    ('2000-06', 0, 25, 'Tech unwind continuing; rotation refuge exists', 'H'),
    ('2000-07', 0, 25, 'Dot-com unwind continuing', 'H'),
    ('2000-08', 0, 25, 'Dot-com unwind continuing', 'H'),
    ('2000-09', 0, 25, 'Dot-com unwind and earnings stress visible', 'H'),
    ('2000-10', 0, 25, 'Dot-com bear market visible', 'H'),
    ('2000-11', 0, 25, 'Dot-com bear market visible', 'H'),
    ('2000-12', 0, 25, 'Dot-com bear market visible; Fed easing expected', 'H'),
    ('2001-01', 0, 25, 'Dot-com recession risk visible; Fed easing capacity high', 'H'),
    ('2001-02', 0, 25, 'Dot-com recession stress continuing', 'H'),
    ('2001-03', 0, 25, 'Dot-com recession stress continuing', 'H'),
    ('2001-04', 0, 25, 'Dot-com recession stress continuing', 'H'),
    ('2001-05', 0, 25, 'Fed-cut cycle active; tech stress continuing', 'H'),
    ('2001-06', 0, 25, 'Fed-cut cycle active; tech stress continuing', 'H'),
    ('2001-07', 0, 25, 'Dot-com recession stress continuing', 'H'),
    ('2001-08', 0, 25, 'Dot-com recession stress continuing', 'H'),
    ('2001-09', 0, 25, 'Month-start dot-com/recession stress; terror attack not yet known', 'H'),
    ('2001-10', 0, 25, 'Post-attack stress visible; Fed response fast and rotation exists', 'H'),
    ('2001-11', 0, 25, 'Post-attack recovery attempt; tech stress still visible', 'H'),
    ('2001-12', 0, 25, 'Recession/dot-com stress still visible', 'H'),
    ('2002-01', 25, 0, 'Dot-com tail; Fed already cutting but slow grind persists', 'H'),
    ('2002-02', 25, 0, 'Dot-com tail with sector rotation refuge', 'H'),
    ('2002-03', 25, 0, 'Dot-com tail with sector rotation refuge', 'H'),
    ('2002-04', 25, 0, 'Dot-com tail with sector rotation refuge', 'H'),
    ('2002-05', 25, 0, 'Accounting-quality concern visible but sector refuge exists', 'H'),
    ('2002-06', 25, 25, 'Accounting scandal / credit concern broadening', 'H'),
    ('2002-07', 25, 25, 'WorldCom/Enron accounting stress visible; rotation still exists', 'H'),
    ('2002-08', 25, 25, 'Accounting and credit stress continuing', 'H'),
    ('2002-09', 25, 25, 'Dot-com tail and accounting stress continuing', 'H'),
    ('2002-10', 25, 25, 'Bear-market stress visible; rotation refuge still exists', 'H'),
    ('2002-11', 25, 0, 'Recovery attempt after dot-com tail stress', 'H'),
    ('2002-12', 25, 0, 'Dot-com tail fading but economy still sluggish', 'H'),
    ('2003-01', 0, 25, 'Iraq-war risk and recession tail visible', 'H'),
    ('2003-02', 0, 25, 'Iraq-war risk and recession tail visible', 'H'),
    ('2003-03', 0, 25, 'Iraq-war risk visible; invasion timing not yet known', 'H'),
    ('2003-04', 0, 25, 'War/recession tail still visible', 'H'),
    ('2003-05', 0, 0, 'Recovery phase; broad stress fading', 'H'),
    ('2003-06', 0, 0, 'Recovery phase; broad stress fading', 'H'),
    ('2003-07', 0, 0, 'Calm recovery', 'H'),
    ('2003-08', 0, 0, 'Calm recovery', 'H'),
    ('2003-09', 0, 0, 'Calm recovery', 'H'),
    ('2003-10', 0, 0, 'Calm recovery', 'H'),
    ('2003-11', 0, 0, 'Calm recovery', 'H'),
    ('2003-12', 0, 0, 'Calm recovery', 'H'),
    ('2004-01', 0, 0, 'Calm expansion', 'H'),
    ('2004-02', 0, 0, 'Calm expansion', 'H'),
    ('2004-03', 0, 0, 'Calm expansion', 'H'),
    ('2004-04', 0, 0, 'Calm expansion', 'H'),
    ('2004-05', 0, 0, 'Calm expansion', 'H'),
    ('2004-06', 0, 0, 'Fed hike cycle expected; no shock catalyst', 'H'),
    ('2004-07', 0, 0, 'Measured Fed hikes; banks healthy', 'H'),
    ('2004-08', 0, 0, 'Measured Fed hikes; banks healthy', 'H'),
    ('2004-09', 0, 0, 'Measured Fed hikes; banks healthy', 'H'),
    ('2004-10', 0, 0, 'Measured Fed hikes; banks healthy', 'H'),
    ('2004-11', 0, 0, 'Measured Fed hikes; banks healthy', 'H'),
    ('2004-12', 0, 0, 'Calm expansion', 'H'),
    ('2005-01', 0, 0, 'Calm expansion; housing froth not yet a crash catalyst', 'H'),
    ('2005-02', 0, 0, 'Calm expansion', 'H'),
    ('2005-03', 0, 0, 'Calm expansion', 'H'),
    ('2005-04', 0, 0, 'Calm expansion', 'H'),
    ('2005-05', 0, 0, 'Calm expansion', 'H'),
    ('2005-06', 0, 0, 'Calm expansion', 'H'),
    ('2005-07', 0, 0, 'Calm expansion', 'H'),
    ('2005-08', 0, 0, 'Calm expansion', 'H'),
    ('2005-09', 0, 0, 'Calm expansion', 'H'),
    ('2005-10', 0, 0, 'Calm expansion', 'H'),
    ('2005-11', 0, 0, 'Calm expansion', 'H'),
    ('2005-12', 0, 0, 'Calm expansion', 'H'),
    ('2006-01', 0, 0, 'Housing froth visible but no funding stress', 'H'),
    ('2006-02', 0, 0, 'Housing froth visible but no funding stress', 'H'),
    ('2006-03', 0, 0, 'Housing slowdown risk building', 'H'),
    ('2006-04', 0, 0, 'Housing slowdown risk building', 'H'),
    ('2006-05', 0, 0, 'Housing slowdown risk building', 'H'),
    ('2006-06', 0, 25, 'Housing slowdown and rate stress visible', 'H'),
    ('2006-07', 0, 25, 'Housing slowdown and subprime concern visible', 'H'),
    ('2006-08', 0, 25, 'Housing/subprime concern visible but contained', 'H'),
    ('2006-09', 0, 25, 'Housing/subprime concern visible but contained', 'H'),
    ('2006-10', 0, 25, 'Housing/subprime concern visible but contained', 'H'),
    ('2006-11', 0, 25, 'Housing/subprime concern visible but contained', 'H'),
    ('2006-12', 0, 25, 'Housing/subprime concern visible but contained', 'H'),
    ('2007-01', 0, 25, 'Subprime deterioration visible but still contained', 'H'),
    ('2007-02', 0, 25, 'Subprime deterioration visible but still contained', 'H'),
    ('2007-03', 25, 25, 'Subprime lender stress visible; banking-adjacent risk rising', 'H'),
    ('2007-04', 25, 25, 'Subprime lender failures visible; credit concern broadening', 'H'),
    ('2007-05', 25, 25, 'Subprime and mortgage-credit stress visible', 'H'),
    ('2007-06', 25, 25, 'Mortgage-credit stress visible; Bear hedge-fund risk building', 'H'),
    ('2007-07', 25, 25, 'Subprime hedge-fund stress visible', 'H'),
    ('2007-08', 25, 50, 'Credit-market contagion visible after mortgage hedge-fund failures', 'H'),
    ('2007-09', 25, 50, 'Global credit crunch visible; Fed easing underway', 'H'),
    ('2007-10', 25, 50, 'Bank writedowns and credit stress visible', 'H'),
    ('2007-11', 25, 50, 'Bank writedowns and funding stress visible', 'H'),
    ('2007-12', 25, 50, 'Credit crunch and recession risk visible', 'H'),
    ('2008-01', 25, 50, 'Credit crisis and recession risk visible; banks under pressure', 'H'),
    ('2008-02', 25, 50, 'Credit crisis and monoline/bank stress visible', 'H'),
    ('2008-03', 25, 50, 'Bank funding stress visible; Bear Stearns failure not yet known', 'H'),
    ('2008-04', 50, 50, 'Post-Bear Stearns: banks are the patient; broad credit stress', 'H'),
    ('2008-05', 50, 50, 'Post-Bear banking stress still unresolved', 'H'),
    ('2008-06', 50, 50, 'Banks are the patient; housing/credit stress universalizing', 'H'),
    ('2008-07', 50, 50, 'Fannie/Freddie and bank stress visible', 'H'),
    ('2008-08', 50, 50, 'Systemic mortgage-credit stress visible', 'H'),
    ('2008-09', 50, 50, 'Pre-Lehman systemic stress visible; rescue path uncertain', 'H'),
    ('2008-10', 50, 50, 'Post-Lehman/AIG/TARP stress; institutional bid net-absent', 'H'),
    ('2008-11', 50, 50, 'GFC panic continuing; banks still the patient', 'H'),
    ('2008-12', 50, 50, 'GFC panic continuing despite policy response', 'H'),
    ('2009-01', 50, 50, 'GFC panic continuing; banks still the patient', 'H'),
    ('2009-02', 50, 50, 'GFC panic continuing; bank nationalization fear visible', 'H'),
    ('2009-03', 50, 50, 'GFC panic visible; market bottom not yet known', 'H'),
    ('2009-04', 25, 25, 'Policy backstops visible; banking stress still elevated', 'H'),
    ('2009-05', 25, 25, 'Stress-test process visible; final relief not yet known', 'H'),
    ('2009-06', 25, 0, 'Post-stress-test healing; banks still recovering', 'H'),
    ('2009-07', 25, 0, 'GFC healing phase; breadth stress contained', 'H'),
    ('2009-08', 25, 0, 'GFC healing phase; breadth stress contained', 'H'),
    ('2009-09', 25, 0, 'GFC healing phase; breadth stress contained', 'H'),
    ('2009-10', 25, 0, 'GFC healing phase; breadth stress contained', 'H'),
    ('2009-11', 25, 0, 'GFC healing phase; breadth stress contained', 'H'),
    ('2009-12', 25, 0, 'GFC healing phase; breadth stress contained', 'H'),
    ('2010-01', 25, 0, 'GFC tail; ~140 small bank failures expected 2010', 'H'),
    ('2010-02', 25, 0, 'Greek concerns rising', 'H'),
    ('2010-03', 25, 0, 'Greek rescue talks; banking still partial', 'H'),
    ('2010-04', 25, 25, 'Month-start Greek rescue talks; SPX late-month peak not yet known', 'H'),
    ('2010-05', 25, 25, 'Month-start Eurozone stress visible; Flash Crash not yet known', 'H'),
    ('2010-06', 25, 25, 'Continued Eurozone stress; VIX 35+', 'H'),
    ('2010-07', 0, 25, 'Recovery starting', 'H'),
    ('2010-08', 0, 0, 'Month-start recovery phase; Jackson Hole QE2 hint not yet known', 'H'),
    ('2010-09', 0, 0, 'QE2 anticipation; banking stable', 'H'),
    ('2010-10', 0, 0, 'QE2 expected; gate applies', 'H'),
    ('2010-11', 0, 0, 'Month-start QE2 expected; launch timing not yet known', 'H'),
    ('2010-12', 0, 0, 'Calm year-end', 'H'),
    ('2011-01', 0, 0, 'Calm; gate applies', 'H'),
    ('2011-02', 0, 0, 'Calm', 'H'),
    ('2011-03', 0, 25, 'Month-start calm; Japan earthquake not yet known', 'H'),
    ('2011-04', 0, 0, 'Aftershocks fading', 'H'),
    ('2011-05', 0, 0, 'QE2 ending soon', 'H'),
    ('2011-06', 0, 25, 'Month-start QE2 end expected', 'H'),
    ('2011-07', 0, 25, 'Italy bond stress builds (mark-to-market cliff)', 'H'),
    ('2011-08', 0, 25, 'Month-start Eurozone/Italy stress visible; US downgrade not yet known', 'H'),
    ('2011-09', 0, 25, 'Eurozone deepening; bank-patient signal', 'H'),
    ('2011-10', 0, 25, 'ECB action begins; calming', 'H'),
    ('2011-11', 0, 25, 'Mario Draghi takes ECB', 'H'),
    ('2011-12', 0, 0, 'Month-start ECB support expected; LTRO not yet known', 'H'),
    ('2012-01', 0, 0, 'LTRO calming Eurozone', 'H'),
    ('2012-02', 0, 0, 'Calm', 'H'),
    ('2012-03', 0, 0, 'Greek default avoided', 'M'),
    ('2012-04', 0, 0, 'Spanish concerns', 'M'),
    ('2012-05', 0, 25, 'Spanish bond stress (bank-adjacent)', 'H'),
    ('2012-06', 0, 25, 'Spanish bank bailout request', 'H'),
    ('2012-07', 0, 0, 'Month-start Spanish stress visible; Draghi pledge not yet known', 'H'),
    ('2012-08', 0, 0, 'OMT announced', 'H'),
    ('2012-09', 0, 0, 'Month-start policy support expected; QE3 not yet known', 'H'),
    ('2012-10', 0, 0, 'Calm', 'H'),
    ('2012-11', 0, 0, 'Calm', 'H'),
    ('2012-12', 0, 0, 'Fiscal cliff fears, minor', 'H'),
    ('2013-01', 0, 0, 'Fiscal cliff resolved', 'H'),
    ('2013-02', 0, 0, 'Calm bull', 'H'),
    ('2013-03', 0, 0, 'Cyprus crisis (sectoral, banking-adjacent EU only)', 'H'),
    ('2013-04', 0, 0, 'Calm', 'H'),
    ('2013-05', 0, 25, 'Month-start calm before taper-tantrum trigger', 'H'),
    ('2013-06', 0, 25, 'Yields spike; EM selloff', 'H'),
    ('2013-07', 0, 25, 'EM stress continues', 'H'),
    ('2013-08', 0, 25, 'EM crisis (India, Indonesia)', 'H'),
    ('2013-09', 0, 0, 'Month-start taper uncertainty; no-taper surprise not yet known', 'H'),
    ('2013-10', 0, 0, 'Govt shutdown Oct 1-16 (risk-off, not Fed)', 'H'),
    ('2013-11', 0, 0, 'Calm', 'H'),
    ('2013-12', 0, 0, 'Month-start taper uncertainty', 'H'),
    ('2014-01', 0, 0, 'EM mini-crisis (Turkey, Argentina)', 'H'),
    ('2014-02', 0, 0, 'EM stress continued', 'H'),
    ('2014-03', 0, 0, 'Yellen first FOMC', 'H'),
    ('2014-04', 0, 0, 'Calm', 'H'),
    ('2014-05', 0, 0, 'Calm', 'H'),
    ('2014-06', 0, 0, 'Oil peaks ~$115', 'H'),
    ('2014-07', 0, 0, 'Calm', 'H'),
    ('2014-08', 0, 0, 'Calm', 'H'),
    ('2014-09', 0, 0, 'Calm', 'H'),
    ('2014-10', 0, 25, 'Month-start rates/oil stress; flash-crash move not yet known', 'H'),
    ('2014-11', 0, 0, 'Recovery', 'H'),
    ('2014-12', 0, 25, 'Oil crashes to $50; SNB stress brewing', 'H'),
    ('2015-01', 0, 25, 'Month-start oil/Europe stress; SNB shock not yet known', 'H'),
    ('2015-02', 0, 0, 'Stabilizing', 'H'),
    ('2015-03', 0, 0, 'Calm', 'H'),
    ('2015-04', 0, 0, 'Calm', 'H'),
    ('2015-05', 0, 0, 'Calm', 'H'),
    ('2015-06', 0, 25, 'Greek 3rd crisis', 'H'),
    ('2015-07', 0, 25, 'China A-shares correcting', 'H'),
    ('2015-08', 0, 25, 'Month-start China/EM stress visible; devaluation not yet known', 'H'),
    ('2015-09', 0, 25, 'EM contagion continues', 'H'),
    ('2015-10', 0, 25, 'Calming', 'H'),
    ('2015-11', 0, 25, 'Dec Fed liftoff approaching', 'H'),
    ('2015-12', 0, 25, 'Month-start Fed liftoff expected; exact hike not yet known', 'H'),
    ('2016-01', 0, 25, 'Oil $26; HY stress (energy sector mark-down)', 'H'),
    ('2016-02', 0, 25, 'Mid-Feb bottom; HY/energy mark-to-market cliff briefly', 'H'),
    ('2016-03', 0, 25, 'Recovery (Yellen dovish)', 'H'),
    ('2016-04', 0, 0, 'Calm', 'H'),
    ('2016-05', 0, 0, 'Calm', 'H'),
    ('2016-06', 0, 25, 'Month-start Brexit referendum risk visible; vote result not yet known', 'H'),
    ('2016-07', 0, 0, 'Post-Brexit recovery', 'H'),
    ('2016-08', 0, 0, 'Calm summer', 'H'),
    ('2016-09', 0, 0, 'Calm', 'H'),
    ('2016-10', 0, 0, 'Pre-election jitters', 'H'),
    ('2016-11', 0, 25, 'Month-start election uncertainty; result not yet known', 'H'),
    ('2016-12', 0, 0, 'Month-start post-election rally; Fed hike expected', 'H'),
    ('2017-01', 0, 0, 'Goldilocks bull', 'H'),
    ('2017-02', 0, 0, 'Calm', 'H'),
    ('2017-03', 0, 0, 'Calm', 'H'),
    ('2017-04', 0, 0, 'Calm', 'H'),
    ('2017-05', 0, 0, 'Calm', 'H'),
    ('2017-06', 0, 0, 'Calm', 'H'),
    ('2017-07', 0, 0, 'Calm', 'H'),
    ('2017-08', 0, 0, 'Charlottesville; North Korea tensions', 'H'),
    ('2017-09', 0, 0, 'Calm', 'H'),
    ('2017-10', 0, 0, 'Calm bull', 'H'),
    ('2017-11', 0, 0, 'Tax cuts approaching', 'H'),
    ('2017-12', 0, 0, 'Month-start tax-cut expectation', 'H'),
    ('2018-01', 0, 0, 'Melt-up', 'H'),
    ('2018-02', 0, 25, 'Month-start volatility fragility visible; XIV event not yet known', 'H'),
    ('2018-03', 0, 25, 'Recovering; trade war begins', 'H'),
    ('2018-04', 0, 0, 'Stabilizing', 'H'),
    ('2018-05', 0, 25, 'Italy political crisis brief', 'H'),
    ('2018-06', 0, 0, 'Trade war escalating', 'H'),
    ('2018-07', 0, 0, 'Calm', 'H'),
    ('2018-08', 0, 25, 'EM crisis (Turkey, Argentina)', 'H'),
    ('2018-09', 0, 0, 'Calm', 'H'),
    ('2018-10', 0, 25, 'Q4 selloff begins; SPX -7%', 'H'),
    ('2018-11', 0, 25, 'Continued selloff', 'H'),
    ('2018-12', 0, 25, 'Month-start Q4 selloff ongoing; Christmas Eve bottom not yet known', 'H'),
    ('2019-01', 0, 25, 'Month-start post-Q4 stress; Powell pivot not yet known', 'H'),
    ('2019-02', 0, 0, 'Recovery', 'H'),
    ('2019-03', 0, 0, 'Yield curve inverts 3M-10Y', 'H'),
    ('2019-04', 0, 0, 'Calm', 'H'),
    ('2019-05', 0, 0, 'Trade war re-escalates', 'H'),
    ('2019-06', 0, 0, 'Fed cut signaled', 'H'),
    ('2019-07', 0, 0, 'Month-start insurance-cut expectation', 'H'),
    ('2019-08', 0, 25, 'Trade war + 2Y/10Y inverts', 'H'),
    ('2019-09', 0, 25, 'Month-start trade/yield-curve stress; repo squeeze not yet known', 'H'),
    ('2019-10', 0, 0, 'Fed standing repo facility', 'H'),
    ('2019-11', 0, 0, 'Trade deal optimism', 'H'),
    ('2019-12', 0, 0, 'Phase 1 trade deal', 'H'),
    ('2020-01', 0, 0, 'Pre-COVID calm', 'H'),
    ('2020-02', 0, 25, 'Month-start COVID emerging; SPX peak not yet known', 'H'),
    ('2020-03', 0, 50, 'Month-start COVID shock visible; March rescue/bottom not yet known', 'H'),
    ('2020-04', 0, 25, 'Recovery starting; QE massive', 'H'),
    ('2020-05', 0, 25, 'V-shape underway; gate restored', 'H'),
    ('2020-06', 0, 0, 'Recovery solid', 'H'),
    ('2020-07', 0, 0, 'Tech leadership', 'H'),
    ('2020-08', 0, 0, 'Bull resumption', 'H'),
    ('2020-09', 0, 0, 'Tech correction; FAANG -10%', 'H'),
    ('2020-10', 0, 0, 'Election approaching', 'H'),
    ('2020-11', 0, 0, 'Month-start election/vaccine uncertainty; results not yet known', 'H'),
    ('2020-12', 0, 0, 'Vaccine rally', 'H'),
    ('2021-01', 0, 0, 'Month-start speculative-excess noise; GameStop squeeze not yet known', 'H'),
    ('2021-02', 0, 0, 'Reflation trade', 'H'),
    ('2021-03', 0, 0, 'Yields rising; rotation', 'H'),
    ('2021-04', 0, 0, 'Calm bull', 'H'),
    ('2021-05', 0, 0, 'Inflation concerns starting', 'H'),
    ('2021-06', 0, 0, 'Month-start inflation/rate concern; hawkish surprise not yet known', 'H'),
    ('2021-07', 0, 0, 'Calm', 'H'),
    ('2021-08', 0, 0, 'Calm', 'H'),
    ('2021-09', 0, 25, 'Evergrande; Chinese property concerns', 'H'),
    ('2021-10', 0, 0, 'Inflation hitting 6%+', 'H'),
    ('2021-11', 0, 0, 'Fed taper acceleration announced', 'H'),
    ('2021-12', 0, 0, 'Inflation 6.8%; rate hikes telegraphed', 'H'),
    ('2022-01', 0, 25, 'Tech selloff; gate ON (banks healthy)', 'H'),
    ('2022-02', 0, 25, 'Month-start Russia/Ukraine risk visible; invasion not yet known', 'H'),
    ('2022-03', 0, 25, 'Month-start Ukraine/oil/inflation stress; Fed hike not yet known', 'H'),
    ('2022-04', 0, 25, 'Crypto + tech selloff intensifying', 'H'),
    ('2022-05', 0, 25, 'Fed 50bp; bonds + stocks both down', 'H'),
    ('2022-06', 0, 25, 'Fed 75bp; crypto crashing (3AC, Celsius)', 'H'),
    ('2022-07', 0, 25, 'Recession fears', 'H'),
    ('2022-08', 0, 25, 'Jackson Hole hawkish', 'H'),
    ('2022-09', 25, 25, 'Month-start rates/UK fragility visible; LDI break not yet known', 'H'),
    ('2022-10', 0, 25, 'BoE backstop ends; tensions', 'H'),
    ('2022-11', 0, 25, 'CPI peaked; rally', 'H'),
    ('2022-12', 0, 0, 'Year-end calming', 'H'),
    ('2023-01', 0, 0, 'Disinflation rally', 'H'),
    ('2023-02', 0, 0, 'Strong jobs data; bull', 'H'),
    ('2023-03', 25, 25, 'Month-start bank-duration risk visible; SVB collapse not yet known', 'H'),
    ('2023-04', 25, 0, 'Month-start regional-bank stress; First Republic resolution not yet known', 'H'),
    ('2023-05', 25, 0, 'Debt ceiling drama', 'H'),
    ('2023-06', 0, 0, 'Resolved; AI rally', 'H'),
    ('2023-07', 0, 0, 'Fed pause near', 'H'),
    ('2023-08', 0, 0, 'Fitch US downgrade Aug 1', 'H'),
    ('2023-09', 0, 0, '10Y > 4.5%', 'H'),
    ('2023-10', 0, 25, 'Month-start rates/geopolitical risk; Mideast war not yet known', 'H'),
    ('2023-11', 0, 0, 'Pivot rally', 'H'),
    ('2023-12', 0, 0, 'Month-start disinflation/pivot expectation; FOMC surprise not yet known', 'H'),
    ('2024-01', 0, 0, 'Calm bull', 'H'),
    ('2024-02', 0, 0, 'AI rally', 'H'),
    ('2024-03', 0, 0, 'Calm', 'H'),
    ('2024-04', 0, 0, 'Inflation re-acceleration', 'H'),
    ('2024-05', 0, 0, 'Calm', 'H'),
    ('2024-06', 0, 0, 'Calm', 'H'),
    ('2024-07', 0, 0, 'Rotation; small-caps surge', 'H'),
    ('2024-08', 0, 25, 'Month-start yen-carry/Nikkei fragility visible; Aug-5 break not yet known', 'H'),
    ('2024-09', 0, 0, 'Month-start Fed-cut expectation', 'H'),
    ('2024-10', 0, 0, 'Pre-election', 'H'),
    ('2024-11', 0, 0, 'Month-start election uncertainty; result not yet known', 'H'),
    ('2024-12', 0, 0, 'Year-end calm', 'H'),
    ('2025-01', 0, 0, 'Month-start policy/AI concentration risk; DeepSeek shock not yet known', 'M'),
    ('2025-02', 0, 0, 'Tariff threats escalating', 'M'),
    ('2025-03', 0, 25, 'Tariff war Mar; SPX correction', 'M'),
    ('2025-04', 0, 25, 'Month-start tariff-policy risk; Apr-2 announcement not yet known', 'M'),
    ('2025-05', 0, 25, 'Tariff pauses; recovery', 'M'),
    ('2025-06', 0, 0, 'Calming', 'M'),
    ('2025-07', 0, 0, 'Calm summer', 'M'),
    ('2025-08', 0, 0, 'AI capex peak narrative', 'M'),
    ('2025-09', 0, 25, 'Fed cuts amid inflation persistence', 'M'),
    ('2025-10', 0, 25, 'AI valuations stretched; capex concerns', 'M'),
    ('2025-11', 25, 25, 'Hyperscaler debt rising; FSB warning brewing', 'M'),
    ('2025-12', 25, 0, 'Year-end AI capex unease', 'M'),
    ('2026-01', 25, 0, 'CPI rising; oil tensions; CRE delinquency record', 'H'),
    ('2026-02', 25, 25, 'Hyperscaler capex guidance up; market wobble', 'H'),
    ('2026-03', 25, 25, 'AI capex stress-test begins', 'H'),
    ('2026-04', 25, 25, 'Month-start AI capex/inflation stress; Apr-30 Meta move not yet known', 'H'),
    ('2026-05', 25, 25, 'Month-start AI capex stress and sector rotation; later FSB/CPI releases not yet known', 'H'),
    ('2026-06', 25, 25, 'Known Apr CPI/PCE inflation; FSB private-credit and AI-valuation risk', 'H'),
]

RISK_REPORT_NOTES_BY_MONTH: dict[str, list[str]] = {
    "2026-06": [
        (
            "Duration remains 25 because inflation and credit/AI stress are "
            "visible, but banks are not yet the obvious patient and a funding "
            "cliff is not already visible at month start."
        ),
        (
            "Breadth remains 25 because the risk is mixed across inflation, "
            "private credit, and AI valuation/capex concentration rather than "
            "universal liquidation stress."
        ),
        (
            "Risk score 50 maps to reduce exposure, not stop entries; the "
            "dashboard gate stays open unless the score reaches 75."
        ),
    ],
}


def classify_recommendation(risk_score: int) -> str:
    """Return the production recommendation for a canonical risk score."""
    if risk_score >= PRODUCTION_STOP_THRESHOLD:
        return "stop"
    if risk_score >= 50:
        return "reduce"
    return "continue"


def validate_rows(rows: list[tuple[str, int, int, str, str]]) -> None:
    """Validate canonical axis values, ordering, and month continuity."""
    allowed_axis_scores = {0, 25, 50}
    previous_year_month = ""
    seen_year_months: set[str] = set()

    for row_tuple in rows:
        year_month, duration_score, breadth_score, _key_event, _confidence = row_tuple
        if year_month in seen_year_months:
            raise ValueError(f"Duplicate year_month: {year_month}")
        seen_year_months.add(year_month)
        if previous_year_month and year_month <= previous_year_month:
            raise ValueError(
                f"Rows must be sorted ascending: {previous_year_month} before {year_month}"
            )
        previous_year_month = year_month
        if duration_score not in allowed_axis_scores:
            raise ValueError(f"Non-canonical duration score for {year_month}")
        if breadth_score not in allowed_axis_scores:
            raise ValueError(f"Non-canonical breadth score for {year_month}")


def write_historical_risk_scores(output_path: Path) -> None:
    """Write the historical risk-score CSV."""
    validate_rows(HISTORICAL_RISK_ROWS)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "year_month",
            "duration_score",
            "breadth_score",
            "risk_score",
            "recommendation",
            "key_event",
            "confidence",
        ])
        for row_tuple in HISTORICAL_RISK_ROWS:
            year_month, duration_score, breadth_score, key_event, confidence = row_tuple
            risk_score = duration_score + breadth_score
            writer.writerow([
                year_month,
                duration_score,
                breadth_score,
                risk_score,
                classify_recommendation(risk_score),
                key_event,
                confidence,
            ])


def build_risk_report_content(
    row_tuple: tuple[str, int, int, str, str],
    generated_date_text: str | None = None,
) -> str:
    """Build a month-specific Markdown risk report.

    Args:
        row_tuple: Risk score tuple in HISTORICAL_RISK_ROWS schema.
        generated_date_text: Optional ISO date used to make tests deterministic.

    Returns:
        Markdown report content for one monthly risk assessment.
    """
    if generated_date_text is None:
        generated_date_text = date.today().isoformat()

    year_month, duration_score, breadth_score, key_event, confidence = row_tuple
    risk_score = duration_score + breadth_score
    recommendation = classify_recommendation(risk_score)
    risk_gate_status = (
        "stop"
        if risk_score >= PRODUCTION_STOP_THRESHOLD
        else "open"
    )
    risk_gate_sentence = (
        "BUY orders are blocked by the dashboard risk-score gate."
        if risk_gate_status == "stop"
        else "BUY orders are not blocked by the dashboard risk-score gate."
    )

    report_lines = [
        f"# Risk Report: {year_month}",
        "",
        f"- Generated date: {generated_date_text}",
        "- Source: scripts/historical_risk_scores.py",
        "- Lookahead rule: use only information visible at month start or earlier.",
        "",
        "## Score",
        "",
        f"- Duration score: {duration_score}",
        f"- Breadth score: {breadth_score}",
        f"- Risk score: {risk_score}",
        f"- Recommendation: {recommendation}",
        f"- Confidence: {confidence}",
        "",
        "## Key event",
        "",
        key_event,
        "",
        "## Assessment notes",
        "",
    ]
    assessment_notes = RISK_REPORT_NOTES_BY_MONTH.get(year_month, [])
    if assessment_notes:
        for assessment_note in assessment_notes:
            report_lines.append(f"- {assessment_note}")
    else:
        report_lines.append("- No expanded notes recorded for this month.")

    report_lines.extend([
        "",
        "## Dashboard gate impact",
        "",
        (
            f"- Gate status: {risk_gate_status} "
            f"(stop threshold: {PRODUCTION_STOP_THRESHOLD})"
        ),
        f"- {risk_gate_sentence}",
        "",
    ])
    return "\n".join(report_lines)


def write_latest_risk_report(
    report_directory: Path,
    generated_date_text: str | None = None,
) -> Path:
    """Write the latest monthly risk report and return its path."""
    validate_rows(HISTORICAL_RISK_ROWS)
    latest_row_tuple = HISTORICAL_RISK_ROWS[-1]
    latest_year_month = latest_row_tuple[0]
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / f"{latest_year_month}.md"
    report_content = build_risk_report_content(
        latest_row_tuple,
        generated_date_text=generated_date_text,
    )
    report_path.write_text(report_content, encoding="utf-8")
    return report_path


def main() -> int:
    """Regenerate the historical CSV and latest monthly risk report."""
    repository_root = Path(__file__).resolve().parent.parent
    output_path = (
        repository_root
        / "data"
        / "historical_risk_scores.csv"
    )
    risk_report_directory = repository_root / "logs" / "risk_report"
    write_historical_risk_scores(output_path)
    report_path = write_latest_risk_report(risk_report_directory)
    recommendation_counts: dict[str, int] = {}
    risk_score_counts: dict[int, int] = {}
    for row_tuple in HISTORICAL_RISK_ROWS:
        _year_month, duration_score, breadth_score, _key_event, _confidence = row_tuple
        risk_score = duration_score + breadth_score
        recommendation = classify_recommendation(risk_score)
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        risk_score_counts[risk_score] = risk_score_counts.get(risk_score, 0) + 1

    print(f"Wrote {len(HISTORICAL_RISK_ROWS)} months to {output_path}")
    print(f"Wrote latest risk report to {report_path}")
    print("Risk score distribution:")
    for risk_score in sorted(risk_score_counts):
        print(f"  {risk_score:3d}: {risk_score_counts[risk_score]:3d} months")
    print("Recommendation distribution:")
    for recommendation in ("continue", "reduce", "stop"):
        month_count = recommendation_counts.get(recommendation, 0)
        print(f"  {recommendation:8s}: {month_count:3d} months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
