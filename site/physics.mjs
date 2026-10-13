/* Independent analytical teaching models. No CFD data enters these functions.
 * r*=r/ell, t*=c0*t/ell, f*=f*ell/c0; ell is an arbitrary reference length.
 * Pressure scale is independently chosen, not calibrated to the cylinder.
 */
export const TAU = 2 * Math.PI;
function positive(value, name) {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be finite and positive`);
}
export function directivity(family, theta, orientation = 0) {
  const angle = theta - orientation;
  if (family === 'monopole') return 1;
  if (family === 'dipole') return Math.cos(angle);
  if (family === 'quadrupole') return Math.cos(2 * angle);
  throw new RangeError('Unknown source family');
}
export function propagation(r, f) {
  positive(r, 'r'); positive(f, 'f');
  return {delay: r, wavelength: 1 / f, kr: TAU * f * r};
}
export function fieldPressure({family = 'dipole', f, r, theta, t, near = false, orientation = 0}) {
  const {kr, delay} = propagation(r, f);
  const phase = TAU * f * (t - delay);
  const reactive = near && family === 'dipole' ? Math.cos(phase) / kr : 0;
  return directivity(family, theta, orientation) * (Math.sin(phase) + reactive) / r;
}
export function loadingSignal(f,t) {
  positive(f,'f');
  return {force:Math.sin(TAU*f*t), derivative:TAU*f*Math.cos(TAU*f*t)};
}
export function dipoleAmplitudes(f,r) {
  const {kr}=propagation(r,f);
  return {radiating:1/r,reactive:1/(kr*r),ratio:1/kr,total:Math.hypot(1,1/kr)/r};
}
export function distributedGain(f,theta,separation) {
  positive(f,'f');
  if(!Number.isFinite(separation)||separation<0)throw new RangeError('Invalid separation');
  // Two equal, coherent point sources at x = ±separation/2, normalized by their compact sum.
  return Math.cos(Math.PI*f*separation*Math.cos(theta));
}
export function coherentPower(count,coherence) {
  if(!Number.isInteger(count)||count<1||!Number.isFinite(coherence)||coherence<0||coherence>1)throw new RangeError('Invalid source count or coherence');
  // Real equal cross-spectral coherence, zero relative phase, equal autospectra of one.
  return count+coherence*count*(count-1);
}
export function aliasFrequency(f,fs) {
  positive(fs,'sample frequency');
  if(!Number.isFinite(f)||f<0)throw new RangeError('Invalid frequency');
  return Math.abs(((f+fs/2)%fs+fs)%fs-fs/2);
}
export function pressureLevel(rmsPa,referencePa=1e-6) {
  positive(referencePa,'reference pressure');
  if(!Number.isFinite(rmsPa)||rmsPa<0)throw new RangeError('Invalid RMS pressure');
  return rmsPa===0?-Infinity:20*Math.log10(rmsPa/referencePa);
}
export function welch(samples,fs,segmentLength=128,window='hann') {
  positive(fs,'sample frequency');
  if(!Array.isArray(samples)||samples.some(x=>!Number.isFinite(x))||!Number.isInteger(segmentLength)||segmentLength<4||segmentLength%2||segmentLength>samples.length||!['hann','boxcar'].includes(window))throw new RangeError('Invalid Welch inputs');
  const n=segmentLength,step=n/2,bins=n/2+1;
  const weights=Array.from({length:n},(_,i)=>window==='hann'?0.5-0.5*Math.cos(TAU*i/n):1);
  const norm=fs*weights.reduce((sum,w)=>sum+w*w,0);
  const frequency=Array.from({length:bins},(_,i)=>i*fs/n),psd=Array(bins).fill(0),spectrogram=Array.from({length:bins},()=>[]),times=[];
  let segments=0;
  for(let start=0;start+n<=samples.length;start+=step){
    const segment=samples.slice(start,start+n),mean=segment.reduce((a,b)=>a+b,0)/n;
    const weighted=segment.map((v,i)=>(v-mean)*weights[i]);
    for(let k=0;k<bins;k++){
      const ca=Math.cos(TAU*k/n),sa=Math.sin(TAU*k/n);
      let cr=1,sr=0,re=0,im=0;
      for(let i=0;i<n;i++){
        re+=weighted[i]*cr;im-=weighted[i]*sr;
        const next=cr*ca-sr*sa;sr=sr*ca+cr*sa;cr=next;
      }
      const density=(re*re+im*im)/norm*((k===0||k===n/2)?1:2);
      psd[k]+=density;spectrogram[k].push(density);
    }
    times.push((start+(n-1)/2)/fs);segments++;
  }
  return {frequency,psd:psd.map(v=>v/segments),spectrogram,times,segments,df:fs/n};
}
