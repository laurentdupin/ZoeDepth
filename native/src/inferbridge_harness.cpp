
#include "inferbridge_harness.h"

#include "zoedepth_native.h"
#include "inferbridge/native_harness_precision.h"
#if defined(ZOEDEPTH_WITH_VULKAN) || defined(ZOEDEPTH_WITH_METAL)
#include "external_gpu.h"
#endif
#if defined(ZOEDEPTH_WITH_METAL)
#include "zoedepth_internal.h"
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

struct ibrh_job;

struct ibrh_runtime {
    std::string error;
    std::string cache_path;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
    bool force_host_transfers = false;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    zoedepth_context* context = nullptr;
    std::string model_path;
    zoedepth_variant variant = ZOEDEPTH_VARIANT_N;
#if defined(ZOEDEPTH_WITH_VULKAN) || defined(ZOEDEPTH_WITH_METAL)
    std::shared_ptr<zoe_native::ExternalGpu> external_gpu;
#endif
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    std::shared_ptr<std::atomic<uint32_t>> occupied_slots=
        std::make_shared<std::atomic<uint32_t>>(0u);
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    std::deque<ibrh_job*> queue;
    bool stopping=false;
    std::thread worker;
#endif
    uint32_t input_size = 384u;
    std::mutex submit_mutex;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
#if defined(ZOEDEPTH_WITH_VULKAN) || defined(ZOEDEPTH_WITH_METAL)
    std::shared_ptr<zoe_native::ExternalJob> gpu_job;
#endif
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    zoe_native::ExternalTextureRequest gpu_request{};
    std::shared_ptr<std::atomic<uint32_t>> occupied_slots;
    std::atomic<uint32_t> state{IBRH_JOB_QUEUED};
    std::atomic<bool> cancel_requested{false};
    std::mutex gpu_mutex;
#endif
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<uint8_t> depth;
};


namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.zoedepth.native";
constexpr char kHarnessVersion[] = "1.2.0";

ibrh_result fail(
    ibrh_runtime* runtime, ibrh_result result, const std::string& message) {
    g_last_error = message;
    if (runtime != nullptr) runtime->error = message;
    return result;
}
std::string copy_string(ibrh_string_view value) {
    return value.size == 0u ? std::string() :
        std::string(value.data, value.size);
}

bool valid_string(ibrh_string_view value) {
    return value.data != nullptr && value.size != 0u &&
        std::memchr(value.data, '\0', value.size) == nullptr;
}

bool json_string(
    const std::string& json, const std::string& key, std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    value = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool json_uint(
    const std::string& json, const std::string& key, uint32_t& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos) return false;
    if (json[position] == '"') ++position;
    size_t end = position;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') ++end;
    if (end == position) return false;
    uint64_t parsed = 0u;
    for (size_t index = position; index < end; ++index) {
        parsed = parsed * 10u + static_cast<uint32_t>(json[index] - '0');
        if (parsed > std::numeric_limits<uint32_t>::max()) return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_luid(const std::string& value, uint64_t& result) {
    if (value.size() != 16u) return false;
    const auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        if (character >= 'A' && character <= 'F') return character - 'A' + 10;
        return -1;
    };
    uint8_t bytes[8]{};
    for (size_t index = 0; index < 8u; ++index) {
        const int high = nibble(value[index * 2u]);
        const int low = nibble(value[index * 2u + 1u]);
        if (high < 0 || low < 0) return false;
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

bool device_index_for_luid(uint64_t luid, int32_t& device_index) {
#if defined(ZOEDEPTH_WITH_VULKAN) && defined(_WIN32)
    for (int32_t index = 0; index < 32; ++index) {
        try {
            const auto capabilities =
                zoe_native::probe_external_gpu(static_cast<uint32_t>(index));
            if (capabilities.available && capabilities.adapter_luid == luid) {
                device_index = index;
                return true;
            }
        } catch (...) {
            if (index == 0) return false;
            break;
        }
    }
#else
    (void)luid;
    (void)device_index;
#endif
    return false;
}

bool input_size(
    const std::string& json, uint32_t fallback, uint32_t& value) {
    value = fallback;
    uint32_t parsed = 0u;
    if (!json_uint(json, "Size", parsed)) return true;
    if (parsed == 0u || parsed > 4096u || parsed % 32u != 0u)
        return false;
    value = parsed;
    return true;
}

zoedepth_variant model_variant(const std::string& parameters) {
    std::string model;
    (void)json_string(parameters, "Models", model);
    if (model == "ZoeK") return ZOEDEPTH_VARIANT_K;
    if (model == "ZoeNK") return ZOEDEPTH_VARIANT_NK;
    return ZOEDEPTH_VARIANT_N;
}

ibrh_result status_result(zoedepth_status status) {
    switch (status) {
        case ZOEDEPTH_STATUS_OK: return IBRH_OK;
        case ZOEDEPTH_STATUS_INVALID_ARGUMENT:
            return IBRH_ERROR_INVALID_ARGUMENT;
        case ZOEDEPTH_STATUS_UNSUPPORTED:
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        case ZOEDEPTH_STATUS_OUT_OF_MEMORY:
        case ZOEDEPTH_STATUS_INTERNAL_ERROR:
        default:
            return IBRH_ERROR_INTERNAL;
    }
}

void retain_job(ibrh_job* job) {
    (void)job->references.fetch_add(1u);
}

void release_job(ibrh_job* job) {
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) {
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
        if(job->occupied_slots)job->occupied_slots->fetch_sub(1u);
#endif
        delete job;
    }
}

#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
void metal_worker_loop(ibrh_model* model){
    for(;;){ibrh_job*job=nullptr;{
        std::unique_lock<std::mutex>lock(model->queue_mutex);
        model->queue_condition.wait(lock,[&]{return model->stopping||
            !model->queue.empty();});
        if(model->stopping&&model->queue.empty())return;
        job=model->queue.front();model->queue.pop_front();}
        if(job->cancel_requested.load()){job->state.store(IBRH_JOB_CANCELLED);
            release_job(job);continue;}
        try{auto native=model->external_gpu->submit_texture(job->gpu_request);
            {std::lock_guard<std::mutex>lock(job->gpu_mutex);
             job->gpu_job=std::move(native);}
            job->state.store(job->cancel_requested.load()?IBRH_JOB_CANCELLED:
                IBRH_JOB_RUNNING);
            if(job->cancel_requested.load())job->gpu_job->cancel();
        }catch(...){job->state.store(IBRH_JOB_FAILED);}
        release_job(job);
    }
}
#endif

ibrh_result IBRH_CALL query_capabilities(
    size_t capabilities_size, ibrh_capabilities* capabilities) {
    if (capabilities == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (capabilities_size < sizeof(*capabilities))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->api_version = IBRH_CURRENT_API_VERSION;
    capabilities->flags = IBRH_CAP_HOST_MEMORY;
    capabilities->input_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->output_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->maximum_inputs = 1u;
    capabilities->maximum_outputs = 1u;
    capabilities->maximum_in_flight_jobs = 1u;
#if defined(ZOEDEPTH_WITH_VULKAN) && defined(_WIN32)
    try {
        if (zoe_native::probe_external_gpu(0u).available) {
            capabilities->flags |=
                IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
                IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
                IBRH_CAP_GPU_RESIDENT_OUTPUT;
            capabilities->input_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->output_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->synchronization_mask =
                1ull << IBRH_SYNC_D3D12_FENCE;
            capabilities->maximum_in_flight_jobs = 3u;
        }
    } catch (...) {
    }
#endif
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    capabilities->flags |=
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    capabilities->input_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->output_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->synchronization_mask =
        1ull << IBRH_SYNC_METAL_SHARED_EVENT;
    capabilities->maximum_in_flight_jobs = 3u;
#endif
    capabilities->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    capabilities->harness_version = {
        kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(
    size_t request_size, const ibrh_runtime_create_request* request,
    ibrh_runtime** output) {
    if (request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto* runtime = new (std::nothrow) ibrh_runtime();
    if (runtime == nullptr) return IBRH_ERROR_INTERNAL;
    runtime->cache_path = copy_string(request->cache_path);
    const std::string device = copy_string(request->requested_device_json);
    std::string transfer_mode;
    runtime->force_host_transfers =
        json_string(device, "transfer_mode", transfer_mode) &&
        transfer_mode == "host";
    uint32_t index = 0u;
    if (json_uint(device, "index", index)) {
        if (index > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "ZoeDepth requested device index is out of range");
        }
        runtime->vulkan_device_index = static_cast<int32_t>(index);
    }
    std::string luid_text;
    if (json_string(device, "luid", luid_text) && !luid_text.empty()) {
        uint64_t luid = 0u;
        if (!parse_luid(luid_text, luid) ||
            !device_index_for_luid(luid, runtime->vulkan_device_index)) {
            delete runtime;
            return fail(nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        "ZoeDepth could not match the requested GPU LUID");
        }
        runtime->adapter_luid = luid;
    }
    *output = runtime;
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) {
    delete runtime;
}

ibrh_result IBRH_CALL model_load(
    ibrh_runtime* runtime, size_t request_size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (runtime == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (!valid_string(request->model_path))
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "ZoeDepth model path is missing");
    const std::string path = copy_string(request->model_path);
    const std::string parameters = copy_string(request->parameters_json);
    inferbridge::native::Precision precision;
    try {
        precision = inferbridge::native::precision_from_parameters_json(parameters);
    } catch (const std::exception& error) {
        return fail(runtime, IBRH_ERROR_INVALID_ARGUMENT, error.what());
    }
    const inferbridge::native::ScopedPrecisionRequest precision_scope(precision);
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    model->model_path = path;
    model->variant = model_variant(parameters);
    if (!input_size(parameters, model->input_size, model->input_size)) {
        delete model;
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "ZoeDepth Size must be a multiple of 32 up to 4096");
    }
#if defined(ZOEDEPTH_WITH_VULKAN) && defined(_WIN32)
    if (runtime->adapter_luid != 0u && !runtime->force_host_transfers) {
        try {
            model->external_gpu = zoe_native::create_external_gpu(
                path, model->variant,
                static_cast<uint32_t>(runtime->vulkan_device_index));
            const auto capabilities = model->external_gpu->capabilities();
            if (!capabilities.available ||
                capabilities.adapter_luid != runtime->adapter_luid)
                throw std::runtime_error(
                    "ZoeDepth loaded on a GPU other than the requested LUID");
        } catch (const std::exception& error) {
            delete model;
            return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY, error.what());
        }
    } else
#endif
    {
        const zoedepth_status status = zoedepth_create_vulkan(
            path.c_str(), model->variant,
            static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
        if (status != ZOEDEPTH_STATUS_OK) {
            const std::string message =
                std::string("ZoeDepth model load failed: ") +
                zoedepth_last_error();
            delete model;
            return fail(runtime, status_result(status), message);
        }
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
        if(!runtime->force_host_transfers){
                try{model->external_gpu=
                    zoe_native::create_metal_external_gpu(
                        model->context, runtime->cache_path);
                model->worker=std::thread(metal_worker_loop,model);
            }catch(const std::exception&error){zoedepth_destroy(model->context);
                delete model;return fail(runtime,
                    IBRH_ERROR_UNSUPPORTED_CAPABILITY,error.what());}
        }
#endif
    }
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    {std::lock_guard<std::mutex>lock(model->queue_mutex);model->stopping=true;
     for(ibrh_job*job:model->queue){job->cancel_requested.store(true);
        job->state.store(IBRH_JOB_CANCELLED);release_job(job);}model->queue.clear();}
    model->queue_condition.notify_all();
    if(model->worker.joinable())model->worker.join();
#endif
#if defined(ZOEDEPTH_WITH_VULKAN) || defined(ZOEDEPTH_WITH_METAL)
    model->external_gpu.reset();
#endif
    zoedepth_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(const ibrh_model* model, size_t size,
    ibrh_model_io_descriptor* out) {
    if (!model || !out) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*out)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *out={}; out->struct_size=sizeof(*out); out->api_version=IBRH_CURRENT_API_VERSION;
    out->input_count=1u; out->output_count=1u; return IBRH_OK;
}
ibrh_result IBRH_CALL model_get_port(const ibrh_model* model, uint32_t direction,
    uint32_t index, size_t size, ibrh_port_descriptor* out) {
    if (!model || !out) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*out)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (index || (direction!=IBRH_PORT_INPUT && direction!=IBRH_PORT_OUTPUT))
        return IBRH_ERROR_NOT_FOUND;
    *out={}; out->struct_size=sizeof(*out); out->api_version=IBRH_CURRENT_API_VERSION;
    out->index=0u; out->direction=direction;
    out->semantic=direction==IBRH_PORT_INPUT?IBRH_SEMANTIC_IMAGE:IBRH_SEMANTIC_DEPTH;
    out->payload_type=direction==IBRH_PORT_INPUT?IBRH_PIXEL_BGRA8:IBRH_PIXEL_DEPTH_FLOAT32;
    out->pixel_format=out->payload_type;
    out->accepted_pixel_format_mask=direction==IBRH_PORT_INPUT?
        ((1ull<<IBRH_PIXEL_BGRA8)|(1ull<<IBRH_PIXEL_RGBA8)):
        (1ull<<IBRH_PIXEL_DEPTH_FLOAT32);
    out->resource_kind=IBRH_RESOURCE_KIND_IMAGE_2D; out->depth=1u;
    out->flags=IBRH_DESCRIPTOR_DYNAMIC_WIDTH|IBRH_DESCRIPTOR_DYNAMIC_HEIGHT;
    return IBRH_OK;
}
ibrh_result IBRH_CALL model_plan_outputs(const ibrh_model* model, size_t size,
    const ibrh_output_plan_request* request, uint32_t capacity,
    ibrh_port_descriptor* outputs) {
    if(!model||!request||!outputs)return IBRH_ERROR_INVALID_ARGUMENT;
    if(size<sizeof(*request)||request->struct_size<sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if(capacity<1u)return IBRH_ERROR_STRUCT_TOO_SMALL;
    if(request->input_count!=1u||!request->inputs||!request->inputs[0].width||
       !request->inputs[0].height)return IBRH_ERROR_INVALID_ARGUMENT;
    auto result=model_get_port(model,IBRH_PORT_OUTPUT,0u,sizeof(outputs[0]),&outputs[0]);
    if(result!=IBRH_OK)return result;
    outputs[0].width=request->inputs[0].width;
    outputs[0].height=request->inputs[0].height; outputs[0].flags=0u;
    return IBRH_OK;
}

ibrh_result IBRH_CALL submit(ibrh_model* model, size_t request_size,
    const ibrh_submit_request* request, ibrh_job** output) {
    if(!model||!request||!output)return IBRH_ERROR_INVALID_ARGUMENT;
    *output=nullptr;
    if(request_size<sizeof(*request)||request->struct_size<sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if(request->input_count!=1u||!request->inputs||
       request->output_count!=1u||!request->outputs)return IBRH_ERROR_INVALID_ARGUMENT;
    const auto& source=request->inputs[0]; const auto& target=request->outputs[0];
    if(source.struct_size<sizeof(source)||target.struct_size<sizeof(target))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    const auto& input=source.resource; const auto& destination=target.resource;
    uint32_t network_size=model->input_size;
    if(!input_size(copy_string(request->parameters_json),network_size,network_size))
        return fail(model->runtime,IBRH_ERROR_INVALID_ARGUMENT,
                    "ZoeDepth Size must be a multiple of 32 up to 4096");
    if(!input.width||!input.height||destination.width!=input.width||
       destination.height!=input.height||
       destination.pixel_format!=IBRH_PIXEL_DEPTH_FLOAT32)
        return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(ZOEDEPTH_WITH_VULKAN) && defined(_WIN32)
    if(input.domain==IBRH_RESOURCE_DOMAIN_D3D12){
        if(!model->external_gpu||destination.domain!=IBRH_RESOURCE_DOMAIN_D3D12||
           input.native_handle_type!=IBRH_NATIVE_HANDLE_WIN32_SHARED||
           destination.native_handle_type!=IBRH_NATIVE_HANDLE_WIN32_SHARED||
           (input.pixel_format!=IBRH_PIXEL_BGRA8&&input.pixel_format!=IBRH_PIXEL_RGBA8)||
           source.synchronization.kind!=IBRH_SYNC_D3D12_FENCE||
           source.synchronization.operation!=IBRH_SYNC_WAIT||
           target.synchronization.kind!=IBRH_SYNC_D3D12_FENCE||
           target.synchronization.operation!=IBRH_SYNC_SIGNAL)
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        auto* job=new(std::nothrow)ibrh_job(); if(!job)return IBRH_ERROR_INTERNAL;
        try{
            std::lock_guard<std::mutex> lock(model->submit_mutex);
            job->gpu_job=model->external_gpu->submit_texture({
                static_cast<uintptr_t>(input.native_handle),input.auxiliary_handle,
                input.width,input.height,
                input.pixel_format==IBRH_PIXEL_RGBA8,network_size,
                static_cast<uintptr_t>(source.synchronization.native_handle),
                source.synchronization.value,
                static_cast<uintptr_t>(destination.native_handle),
                destination.auxiliary_handle,destination.width,destination.height,
                static_cast<uintptr_t>(target.synchronization.native_handle),
                target.synchronization.value,
                request->source_frame_id,request->timestamp_ns});
        }catch(const std::invalid_argument& e){delete job;return fail(model->runtime,IBRH_ERROR_INVALID_ARGUMENT,e.what());
        }catch(const std::exception& e){delete job;return fail(model->runtime,IBRH_ERROR_UNSUPPORTED_CAPABILITY,e.what());}
        job->source_frame_id=request->source_frame_id;job->timestamp_ns=request->timestamp_ns;
        job->width=input.width;job->height=input.height;*output=job;return IBRH_OK;
    }
#endif
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    if(input.domain==IBRH_RESOURCE_DOMAIN_METAL){
        const auto&wait=source.synchronization;
        const auto&signal=target.synchronization;
        const bool no_wait=wait.kind==IBRH_SYNC_NONE;
        const bool event_wait=wait.kind==IBRH_SYNC_METAL_SHARED_EVENT&&
            wait.operation==IBRH_SYNC_WAIT&&
            wait.native_handle_type==IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT&&
            wait.native_handle!=0u;
        if(!model->external_gpu||destination.domain!=IBRH_RESOURCE_DOMAIN_METAL||
           input.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_TEXTURE||!input.native_handle||
           destination.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_TEXTURE||!destination.native_handle||
           (input.pixel_format!=IBRH_PIXEL_BGRA8&&input.pixel_format!=IBRH_PIXEL_RGBA8)||
           (!no_wait&&!event_wait)||signal.kind!=IBRH_SYNC_METAL_SHARED_EVENT||
           signal.operation!=IBRH_SYNC_SIGNAL||
           signal.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT||
           !signal.native_handle||!signal.value)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        uint32_t occupied=model->occupied_slots->load();
        while(occupied<3u&&!model->occupied_slots->compare_exchange_weak(
            occupied,occupied+1u)){}
        if(occupied>=3u)return IBRH_ERROR_INVALID_STATE;
        auto*job=new(std::nothrow)ibrh_job();
        if(!job){model->occupied_slots->fetch_sub(1u);return IBRH_ERROR_INTERNAL;}
        job->occupied_slots=model->occupied_slots;
        job->source_frame_id=request->source_frame_id;
        job->timestamp_ns=request->timestamp_ns;
        job->width=input.width;job->height=input.height;
        job->gpu_request={static_cast<uintptr_t>(input.native_handle),
            input.auxiliary_handle,input.width,input.height,
            input.pixel_format==IBRH_PIXEL_RGBA8,network_size,
            event_wait?static_cast<uintptr_t>(wait.native_handle):0u,
            event_wait?wait.value:0u,
            static_cast<uintptr_t>(destination.native_handle),
            destination.auxiliary_handle,destination.width,destination.height,
            static_cast<uintptr_t>(signal.native_handle),signal.value,
            request->source_frame_id,request->timestamp_ns};
        {std::lock_guard<std::mutex>lock(model->queue_mutex);
         if(model->stopping){release_job(job);return IBRH_ERROR_INVALID_STATE;}
         retain_job(job);model->queue.push_back(job);}
        model->queue_condition.notify_one();*output=job;return IBRH_OK;
    }
#endif
    if(input.domain!=IBRH_RESOURCE_DOMAIN_HOST||destination.domain!=IBRH_RESOURCE_DOMAIN_HOST||
       input.native_handle_type!=IBRH_NATIVE_HANDLE_HOST_POINTER||
       destination.native_handle_type!=IBRH_NATIVE_HANDLE_HOST_POINTER||
       input.pixel_format!=IBRH_PIXEL_BGRA8||
       source.synchronization.kind!=IBRH_SYNC_NONE||
       target.synchronization.kind!=IBRH_SYNC_NONE)
        return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
    const auto* bgra=reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(input.native_handle))+input.byte_offset;
    auto* depth=reinterpret_cast<float*>(
        static_cast<uintptr_t>(destination.native_handle)+destination.byte_offset);
    zoedepth_status status;
    {std::lock_guard<std::mutex> lock(model->submit_mutex);
     status=zoedepth_infer_bgra8_f32(model->context,bgra,input.row_stride_bytes,
        static_cast<int32_t>(input.width),static_cast<int32_t>(input.height),
        static_cast<int32_t>(network_size),depth,
        static_cast<size_t>(input.width)*input.height);}
    if(status!=ZOEDEPTH_STATUS_OK)
        return fail(model->runtime,status_result(status),zoedepth_last_error());
    auto* job=new(std::nothrow)ibrh_job();if(!job)return IBRH_ERROR_INTERNAL;
    job->source_frame_id=request->source_frame_id;job->timestamp_ns=request->timestamp_ns;
    job->width=input.width;job->height=input.height;*output=job;return IBRH_OK;
}

ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    std::shared_ptr<zoe_native::ExternalJob> native_job;
    {std::lock_guard<std::mutex>lock(
        const_cast<ibrh_job*>(job)->gpu_mutex);native_job=job->gpu_job;}
    if(native_job){
        switch(native_job->state()){
            case zoe_native::ExternalJobState::running:
                status->state = IBRH_JOB_RUNNING; break;
            case zoe_native::ExternalJobState::complete:
                status->state = IBRH_JOB_COMPLETE; break;
            case zoe_native::ExternalJobState::cancelled:
                status->state = IBRH_JOB_CANCELLED; break;
        }
    }else status->state=job->state.load();
#elif defined(ZOEDEPTH_WITH_VULKAN)
    if (job->gpu_job) {
        switch (job->gpu_job->state()) {
            case zoe_native::ExternalJobState::running:
                status->state = IBRH_JOB_RUNNING; break;
            case zoe_native::ExternalJobState::complete:
                status->state = IBRH_JOB_COMPLETE; break;
            case zoe_native::ExternalJobState::cancelled:
                status->state = IBRH_JOB_CANCELLED; break;
        }
    } else status->state = IBRH_JOB_COMPLETE;
#else
    status->state = IBRH_JOB_COMPLETE;
#endif
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(ZOEDEPTH_WITH_METAL) && defined(__APPLE__)
    job->cancel_requested.store(true);
    std::shared_ptr<zoe_native::ExternalJob> native_job;
    {std::lock_guard<std::mutex>lock(job->gpu_mutex);native_job=job->gpu_job;}
    if(native_job){native_job->cancel();return IBRH_OK;}
    const uint32_t state=job->state.load();
    if(state==IBRH_JOB_QUEUED||state==IBRH_JOB_RUNNING)return IBRH_OK;
#elif defined(ZOEDEPTH_WITH_VULKAN)
    if (job->gpu_job) {
        job->gpu_job->cancel();
        return IBRH_OK;
    }
#endif
    return IBRH_ERROR_INVALID_STATE;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
}

ibrh_result IBRH_CALL get_last_error(
    const void* object, char* destination, size_t destination_size,
    size_t* required_size) {
    const auto* runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message =
        runtime != nullptr && !runtime->error.empty() ?
        runtime->error : g_last_error;
    const size_t required = message.size() + 1u;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* api) {
    if (api == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (api_size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_api_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload = model_unload;
    api->model_describe_io = model_describe_io;
    api->model_get_port = model_get_port;
    api->model_plan_outputs = model_plan_outputs;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
