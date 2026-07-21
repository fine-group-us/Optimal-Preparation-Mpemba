      program dsmc
c     Simulate a granular gas with inelastic collisions, in contact with a thermal bath at Ts.
c     Every stepkick collisions, the molecules are kicked with a gaussian force that represents the bath         
c     The molecules have mass m=1 and diameter 1. The restitution coeffient is alpha
c     We take k_B=1
c     T0=initial temperature
c     Temp=Gas temperature
c     Ts=bath temperature
c     Tmin= Minimum bath temperature
c     Tmax= Maximum bath temperature
c     N=number of particles
c     v(N,d)= velocity in d dimensions
c     vcm(d)=velocity of the mass center
c     Deltat=unit of time
c     chi=variance of the stochastic white noise force.
c     a2= kurtosis
c     tau=scaled adimensional time
c     tauf=final time
c     ntray=number of trajectories for averaging
      
      
      implicit none  
    
      integer i,j,id,N,d,iseed,ncol,stepkick,nkick,nt,ntray,ntime
      integer npoints

      real ran3
    
      real*8 vmod,varg,pi,rN,x,y,alpha,q
      real*8 phi,scprod,scprodmax,prob
      real*8 time,a2s,z0,chi,a2HCS
      real*8 timeold,timeinterval,val
      real*8 v2kk,Ts,v2,v4,a2,temp,z
      real*8 Deltat,tau,taueq,T0,Xmin,Tmin,Dt,B,ctrl,tauf
      
      parameter (alpha=0.9d0,T0=1.d0,Xmin=1.0d-4,npoints=20)
      parameter (Tmin=Xmin**(2.d0/3.d0))
      parameter (N=10**6,d=3,tauf=1.558d0,Dt=tauf/dfloat(npoints))
      parameter (stepkick=N/2000,taueq=6.d0,ntray=20)
	
      double precision v(N,d),vcm(d),sigma(d)
      double precision T(0:npoints),kur(0:npoints)
      
      open(30,file='Bang_09a_1d-4Xmin_1d6N_20prom.dat',status="new")
      
!     Initialize some parameters
      pi=4.d0*datan(1.d0)
      rN=dble(N)          
      iseed=-123456
      a2s=16.d0*(1.d0-alpha)*(1.d0-2.d0*alpha**2)
      q=73.d0+56.d0*d-24.d0*d*alpha-105.d0*alpha
      q=q+30.d0*(1.d0-alpha)*alpha**2
      a2s=a2s/q
      a2HCS=16.d0*(1.d0-alpha)*(1.d0-2.d0*alpha**2)
      q=25.d0+24.d0*d+alpha*(8.d0*d-57.d0)+2.d0*alpha**2*(alpha-1.d0)
      a2HCS=a2HCS/q
      B=a2HCS/(a2HCS-a2s)
      if (d.eq.2) then 
        z0=(1.d0-alpha**2)*dsqrt(pi)
      else 
        z0=4.d0*(1.d0-alpha**2)*dsqrt(pi)/3.d0 
      endif

      a2=0.d0
      temp=0.d0
      do 1 nt=1,ntray
c     write(2,*) nt
      scprodmax=10.7d0
      Deltat=(4.d0-d)/pi/rN/scprodmax
      ctrl=Dt
      ncol=0
      nkick=0
      time=0.d0
      timeold=0.d0
      ntime=0
      
!---------------------------------------------------                               
!     Gaussian velocity distribution with temperature=T0
!------------------------------------------------------------
      do i=1,N
13      continue        
        x=ran3(iseed)
        if (x.eq.0) goto 13         
        y=ran3(iseed)
        vmod=dsqrt(-2.d0*T0*dlog(x))
        varg=2.d0*pi*y
        v(i,1)=vmod*dcos(varg)
        v(i,2)=vmod*dsin(varg)
14      x=ran3(iseed)
        if (x.eq.0) goto 14         
        y=ran3(iseed)
        vmod=dsqrt(-2.d0*T0*dlog(x))
        varg=2.d0*pi*y
        v(i,3)=vmod*dcos(varg)
      enddo   
      
!----------------------------------------------------      
!     Prepare the initial stationary state 
!----------------------------------------------------  
      do 99 while (tau.lt.taueq)
        chi=z0*T0**1.5d0*(1.d0+3.d0/16.d0*a2s)
!     We choose a random direction sigma in d-dimensions     
        x=ran3(iseed)
        phi=2.d0*pi*x
        sigma(1)=dcos(phi)
        sigma(2)=dsin(phi)
        if (d.eq.3) then
54       x=ran3(iseed)
         if (x.gt.1.d0) go to 54
         x=2.d0*x-1.d0
         y=dsqrt(1.d0-x**2)
         sigma(1)=sigma(1)*y
         sigma(2)=sigma(2)*y
         sigma(3)=x       
        endif     
         
!     We choose the pair to collide: there is always a collision, because either
!     (i,j) or (j,i) verifies that the scalar product of the relative velocity with
!     sigma is positive 
       i=1+int(N*ran3(iseed))
10     continue
       j=1+int(N*ran3(iseed))      
       if (j.eq.i) goto 10
       if (i.gt.N) i=N
       if (j.gt.N) j=N
       scprod=0.d0
       do id=1,d
         scprod=scprod+(v(i,id)-v(j,id))*sigma(id)
       enddo
                  
!     Time is increased either a collision is accepted or not
       time=time+deltat
     
 !    Accept the collision with a probability proportional to the scalar product
 !    of its relative velocity and change the velocities
       x=ran3(iseed)
       prob=dabs(scprod)/scprodmax
       if (dabs(scprod).gt.scprodmax) then 
           write(1,199) T0,scprod
           scprodmax=dabs(scprod)
           Deltat=(4.d0-d)/pi/rN/scprodmax
       endif
       if (x.lt.prob) then
            do id=1,d
                v(i,id)=v(i,id)-(1.d0+alpha)*scprod*sigma(id)/2.d0
                v(j,id)=v(j,id)+(1.d0+alpha)*scprod*sigma(id)/2.d0
            enddo 

!     updates the collision counters
         ncol=ncol+1 
	 nkick=1       
       endif
                  
!     Every given number of collisions, we kick all the particles at random
        if (mod(ncol,stepkick).eq.0 .and.(nkick.eq.1)) then
          timeinterval=time-timeold
          val=sqrt(chi*timeinterval)

          do i=1,N
15          continue
            x=ran3(iseed)
            if (x.eq.0.d0) goto 15
            y=ran3(iseed)
            v(i,1)=v(i,1)+val*dsqrt(-2.d0*dlog(x))*dcos(2.d0*pi*y)
            v(i,2)=v(i,2)+val*dsqrt(-2.d0*dlog(x))*dsin(2.d0*pi*y)
            if (d.eq.3) then
16            continue
              x=ran3(iseed)
              if (x.eq.0.d0) goto 16
              y=ran3(iseed)
              v(i,3)=v(i,3)+val*dsqrt(-2.d0*dlog(x))*dcos(2.d0*pi*y)
            endif
          enddo

	  nkick=0
          timeold=time
        endif
        tau=z0*dsqrt(T0)*time
99     continue 
     
c     Initial Temperature after eliminating a nonvanishing CM velocity
      do id=1,d
         vcm(id)=0.d0
         do i=1,N
            vcm(id)=vcm(id)+v(i,id)
         enddo
         vcm(id)=vcm(id)/rN
      enddo
      
      v2=0.d0
      v4=0.d0
      do i=1,N
            v2kk=0.d0
            do id=1,d
              v(i,id)=v(i,id)-vcm(id)
              v2kk=v2kk+v(i,id)**2.d0            
            enddo
            v2=v2+v2kk
            v4=v4+v2kk**2
      enddo        
      v2=v2/rN
      temp=v2/d
      v4=v4/rN
      a2=v4/temp**2.d0/dble(d*(d+2))-1.d0

      T(0)=temp+T(0)
      kur(0)=kur(0)+a2
      
!----------------------------------------------------      
!     Bang 
!---------------------------------------------------- 
      time=0.d0
      tau=0.d0
      ncol=0
      nkick=0
      timeold=0.d0
      Ts=Tmin
      do 100 while (tau.lt.tauf) 
       chi=z0*Ts**1.5d0*(1.d0+3.d0/16.d0*a2s)

!     We choose a random direction sigma in d-dimensions     
        x=ran3(iseed)
        phi=2.d0*pi*x
        sigma(1)=dcos(phi)
        sigma(2)=dsin(phi)
        if (d.eq.3) then
55       x=ran3(iseed)
         if (x.gt.1.d0) go to 55
         x=2.d0*x-1.d0
         y=dsqrt(1.d0-x**2)
         sigma(1)=sigma(1)*y
         sigma(2)=sigma(2)*y
         sigma(3)=x       
        endif     
         
!     We choose the pair to collide: there is always a collision, because either
!     (i,j) or (j,i) verifies that the scalar product of the relative velocity with
!     sigma is positive 
       i=1+int(N*ran3(iseed))
11     continue
       j=1+int(N*ran3(iseed))      
       if (j.eq.i) goto 11
       if (i.gt.N) i=N
       if (j.gt.N) j=N
       scprod=0.d0
       do id=1,d
         scprod=scprod+(v(i,id)-v(j,id))*sigma(id)
       enddo
                  
!     Time is increased either a collision is accepted or not
       time=time+deltat
       tau=z0*dsqrt(T0)*time
       
!    Accept the collision with a probability proportional to the scalar product
!    of its relative velocity and change the velocities
       x=ran3(iseed)
       prob=dabs(scprod)/scprodmax
       if (dabs(scprod).gt.scprodmax) then 
           write(1,*) Ts,scprod
           scprodmax=dabs(scprod)
           Deltat=(4.d0-d)/pi/rN/scprodmax
       endif
       if (x.lt.prob) then
            do id=1,d
                v(i,id)=v(i,id)-(1.d0+alpha)*scprod*sigma(id)/2.d0
                v(j,id)=v(j,id)+(1.d0+alpha)*scprod*sigma(id)/2.d0
            enddo 
!     updates the collision counters
         ncol=ncol+1 
	 nkick=1       
       endif
                  
!     Every given number of collisions, we kick all the particles at random
        if (mod(ncol,stepkick).eq.0 .and.(nkick.eq.1)) then
          timeinterval=time-timeold
          val=sqrt(chi*timeinterval)
          do i=1,N
5           continue
            x=ran3(iseed)
            if (x.eq.0.d0) goto 5
            y=ran3(iseed)
            v(i,1)=v(i,1)+val*dsqrt(-2.d0*dlog(x))*dcos(2.d0*pi*y)
            v(i,2)=v(i,2)+val*dsqrt(-2.d0*dlog(x))*dsin(2.d0*pi*y)
            if (d.eq.3) then
6             continue
              x=ran3(iseed)
              if (x.eq.0.d0) goto 6
              y=ran3(iseed)
              v(i,3)=v(i,3)+val*dsqrt(-2.d0*dlog(x))*dcos(2.d0*pi*y)
            endif
          enddo
	  nkick=0
          timeold=time
        endif
     
c     Compute granular temperature every Dt
        if (tau.ge.ctrl) then 
          do id=1,d
            vcm(id)=0.d0
            do i=1,N
            vcm(id)=vcm(id)+v(i,id)
            enddo
            vcm(id)=vcm(id)/rN
          enddo

          ntime=ntime+1
          v2=0.d0
          v4=0.d0
          do i=1,N
            v2kk=0.d0
            do id=1,d
              v(i,id)=v(i,id)-vcm(id)
              v2kk=v2kk+v(i,id)**2.d0            
            enddo
            v2=v2+v2kk
            v4=v4+v2kk**2
          enddo        
          v2=v2/rN
          temp=v2/d
          v4=v4/rN
          a2=v4/temp**2.d0/dble(d*(d+2))-1.d0

          T(ntime)=temp+T(ntime)
          kur(ntime)=kur(ntime)+a2
          
          ctrl=ctrl+Dt
          
        endif
          
100   continue

1     continue      

c     Final result
      z=dfloat(ntray)
      do i=0,npoints
         tau=i*Dt
         write(30,199) tau,T(i)/z,kur(i)/z
      enddo
         
      close(30)
199   format(10(e14.7,2x))
      
      stop
      end
       
c====================================================================
!     Subrutina Ran3 numeros aleatorios	
C====================================================================
C       PROGRAM: ran3.f
C       TYPE   : function
C       PURPOSE: generate random numbers
C       VERSION: 17 June 94
C       COMMENT: Initialize idum with negative integer
C======================================================================
        real function ran3(idum)
        Parameter (mbig=1000000000,Mseed=161803398,Mz=0,fac=1./Mbig)
        Dimension MA(55)
        save
        if (idum.lt.0.or.iff.eq.0) then
	   iff=1
	   mj=mseed-iabs(idum)
	   mj=mod(mj,mbig)
	   ma(55)=mj
	   mk=1
	   do 11 i=1,54
	      ii=mod(21*i,55)
	      ma(ii)=mk
	      mk=mj-mk
	      if (mk.lt.mz) mk=mk+mbig
	      mj=ma(ii)
 11	   continue
	   do 13 k=1,4
	      do 12 i=1,55
		 ma(i)=ma(i)-ma(1+mod(i+30,55))
		 if (ma(i).lt.mz) ma(i)=ma(i)+mbig
 12	      continue
 13	   continue
	   inext=0
	   inextp=31
	   idum=1
	end if
	inext=inext+1
	if (inext.eq.56) inext=1
	inextp=inextp+1
	if (inextp.eq.56) inextp=1
	mj=ma(inext)-ma(inextp)
	if (mj.lt.mz) mj=mj+mbig
	ma(inext)=mj
	ran3=mj*fac
	return
	end











    
    
