from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from services.observability_service import MetricsService

router = APIRouter(tags=['Health'])


@router.get('/health', summary='Health Check', description='Returns the current health status of the API. Used by load balancers and uptime monitors.', responses={200: {'description': 'Service is healthy'}})
async def health_check():
                                                 
    return {'status': 'ok', 'message': 'API is healthy and running.'}


@router.get('/metrics', response_class=PlainTextResponse, summary='Prometheus Metrics')
async def metrics():
    return MetricsService.render_prometheus()
