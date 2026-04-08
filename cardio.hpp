#include <cstdint>
#include <vector>

struct HeartBeat
{
    uint32_t timestamp_ms;
    uint32_t rr_interval_ns;
};

struct HRVStatus
{
    int mean_bpm;
    float SDNN;
    float RMSSD;
    float NN50;
    float pNN50;
    int total_beats;
};

class HeartRateAnalyzer
{
    HeartRateAnalyzer() = default;

    int add_data(std::vector<float> data);
    int push_data(float sample, uint32_t timestamp_ms);

    HRVStatus get_status(uint32_t window_start_ms, uint32_t window_end_ms);
    std::vector<HeartBeat> get_beats(uint32_t window_start_ms, uint32_t window_end_ms);
};