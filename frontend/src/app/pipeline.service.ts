import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PipelineService {

  constructor(private http: HttpClient) {}

  sendMessage(message: string): Observable<any> {
    return this.http.post<any>('/chat', { message });
  }

  predict(payload: any): Observable<any> {
    return this.http.post<any>('/predict', payload);
  }

  getExperiments(): Observable<any> {
    return this.http.get<any>('/experiments');
  }

  triggerDriftCheck(): Observable<any> {
    return this.http.post<any>('/drift', {});
  }

  checkHealth(): Observable<any> {
    return this.http.get<any>('/health');
  }
}
