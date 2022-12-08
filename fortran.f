function sbessi (l,x)
    implicit double precision (a-h,o-z)
c
c     ------------------------------------------------------------------
c     Scaled Modified Spherical Bessel function s = exp(-x)*i_l(x).
c     ------------------------------------------------------------------
c
c     Special cases
c
    if (x .le. 0.d0) then
        if (l .eq. 0) then
          s = 1.d0
        else
          s = 0.d0
        endif
        go to 1
    endif
    twox = 2*x
    on2x = 1.d0/twox
    if (x .lt. 19.d0) then
        s = (1.d0-exp(-twox))*on2x
    else
        s = on2x
    endif
    if (l .eq. 0) go to 1
    s0 = s
c
c     Asymptotic formula for large x
c
    rmu = (l+0.5d0)**2
    if (x .gt. max(19.d0,rmu)) then
        s = 1.d0
        t = 1.d0
        u = 0.5d0
        v = twox
        do k = 1,l
          t = -(rmu-u*u)*t/v
          s = s+t
          u = u+1.d0
          v = v+twox
        enddo
        s = on2x*s
        go to 1
    endif
c
c     Downward recursion for small x
c
    m = l+10+int(x)
    u = 4*on2x
    t = (m+1.5d0)*u
    sp = 0.d0
    s = 1.d0
    do n = m,l+1,-1
        t = t-u
        sn = t*s+sp
        sp = s/sn
        s = 1.d0
    enddo
    do n = l,1,-1
        t = t-u
        sn = t*s+sp
        sp = s
        s = sn
        if (s .gt. 1.d+20) then
          s = 0.d0
          go to 1
        endif
    enddo
    s = s0/s
  1  sbessi = s
    return
    end
