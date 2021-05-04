package scalation.analytics
package forecaster

import scala.collection.mutable.Set
import scala.math.{max, min}

import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD}
import scalation.math.noDouble
import scalation.plot.Plot
import scalation.random.{Normal, Random}
import scalation.stat.{Statistic, vectorD2StatVector}
import scalation.util.banner

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Homework 17 : Question 3
** Use the ScalaTion class AR to develop Auto-Regressive Models for p = 1,2,3, for the Lake Level Time-series Dataset. Plot yˆt and yt versus t for each model. 
** p is equal to hyper parameter 
** equation for 0 centered AR model included past p values 
** zt = φ0zt−1 + φ1zt−2 + ... + φp−1zt−p + εt
**
** the forcasted value for one step ahead(h=1) is calculated by 
** zˆ=φz +φz +...+φp−1 t−p

*/


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



object ARHmwk extends App
{

//data for Lake Level Time - Series was found in Forcaster class

val t = VectorD.range (0, 98)
val y = VectorD (580.38, 581.86, 580.97, 580.80, 579.79, 580.39, 580.42, 580.82, 581.40, 581.32,
                     581.44, 581.68, 581.17, 580.53, 580.01, 579.91, 579.14, 579.16, 579.55, 579.67,
                     578.44, 578.24, 579.10, 579.09, 579.35, 578.82, 579.32, 579.01, 579.00, 579.80,
                     579.83, 579.72, 579.89, 580.01, 579.37, 578.69, 578.19, 578.67, 579.55, 578.92,
                     578.09, 579.37, 580.13, 580.14, 579.51, 579.24, 578.66, 578.86, 578.05, 577.79,
                     576.75, 576.75, 577.82, 578.64, 580.58, 579.48, 577.38, 576.90, 576.94, 576.24,
                     576.84, 576.85, 576.90, 577.79, 578.18, 577.51, 577.23, 578.42, 579.61, 579.05,
                     579.26, 579.22, 579.38, 579.10, 577.95, 578.12, 579.75, 580.85, 580.41, 579.96,
                     579.61, 578.76, 578.18, 577.21, 577.13, 579.10, 578.25, 577.91, 576.89, 575.96,
                     576.80, 577.68, 578.38, 578.52, 579.74, 579.31, 579.89, 579.96)

    var ar: AR = null                                          // initalize AR variable as null
    for (h <- 1 to 1) {                                        // creating a for look from h, which is number of steps ( h = 1)
        for (p <- 1 to 3) {                                   // creating a for loop to assign hyper-parameter p to 1 ,2 ,3
            ARMA.hp("p") = p                                  // reassigns hyper-parameter p 
            banner (s"HMWK 17 Lake Levels : AR ($p) with h = $h")
            ar = new AR (y)                                    // time series data
            ar.train (null, y).eval ()                         // train for AR(p) model
            println (ar.report)
            val yf0 = ar.predictAll ()
            val yf1 = ar.forecastAll (h=h, p)(h)              /// make forecasts for all (h=1)
            new Plot (t, y, yf0, s"predictAll: Plot of y, AR($p) vs. t", true)
//          new Plot (t, y, yf1, s"forecastAll: Plot of y, AR($p) vs. t", true)
            assert (yf0 == yf1)
        } // for
    } // for

    val yf = new VectorD (y.dim)                               // test h = 1
    for (t <- yf.range) yf(t) = ar.forecast (t, 1)(0)
    new Plot (t, y, yf, s"forecast (h=1): Plot of y, AR(p) vs. t", true) //makes forcasts for h =1

    val yf2 = new VectorD (y.dim)                              // test h = 2
    for (t <- yf.range) yf2(t) = ar.forecast (t, 2)(1)
    new Plot (t, y, yf2, s"forecast (h=2): Plot of y, AR(p) vs. t", true) //makes forecasts for h = 2


    banner ("Stat Table")
    val stats = SimpleRollingValidation.crossValidate2 (ar, kt_ = 5) //creates table
    Fit.showQofStatTable (stats) //prints table

} 


