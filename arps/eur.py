arps.t.el <- function(decl, q.limit)
{
    min.time <- if (is(decl, 'buildup'))
        decl$time.to.peak
    else
        0
    nlminb(10,   # initial guess 10 periods
            function (t) ((arps.q(decl, t) - q.limit) ^ 2), # cost function
            lower=min.time,   # minimum 0 or time-to-peak
            upper=1e6  # 1 million [time units]
    )$par
}

arps.eur <- function(decl, q.limit)
{
    arps.Np(decl, arps.t.el(decl, q.limit))
}

def arps_eur(decline_rate, q_limit,**kwargs):
