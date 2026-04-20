#include <cmath>
#include <cstddef>
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
    std::vector<HeartBeat> _pam_tompkins(std::vector<float>& raw_data_mv, unsigned int sampling_rate_hz);

    public:
        HeartRateAnalyzer() = default;

        int add_data(std::vector<float> data);
        int push_data(float sample, uint32_t timestamp_ms);

        HRVStatus get_status(uint32_t window_start_ms, uint32_t window_end_ms);
        std::vector<HeartBeat> get_beats(uint32_t window_start_ms, uint32_t window_end_ms);
};

// based on https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm
std::vector<HeartBeat> HeartRateAnalyzer::_pam_tompkins(std::vector<float>& raw_data_mv, const unsigned int sampling_rate_hz=200){
    std::vector<float> lowpass(raw_data_mv.size(), 0.0f); // TODO: optimization (circular buffer or whatever)

    for (size_t i = 12; i < raw_data_mv.size(); i++){
        lowpass[i] = raw_data_mv[i] 
            - 2*raw_data_mv[i-6] 
            + raw_data_mv[i-12] 
            + 2*lowpass[i-1]
            - lowpass[i-2];
        lowpass[i] /= 32; // normalize gain
    }

    std::vector<float> highpass(lowpass.size(), 0.0f);

    for (size_t i = 32; i < lowpass.size(); i++){
        highpass[i] = highpass[i-1] 
            - lowpass[i]/32.0f 
            + lowpass[i-16] 
            - lowpass[i-17] 
            + lowpass[i-32]/32.0f;
    }

    std::vector<float> derivative(highpass.size(), 0.0f);

    for (size_t i = 4; i < highpass.size(); i++){
        derivative[i] = 0.125 * (2*highpass[i] 
                            + highpass[i-1] 
                            - highpass[i-3] 
                            - 2*highpass[i-4]);
    }

    std::vector<float> squared(derivative.size(), 0.0f);

    for (size_t i = 0; i < derivative.size(); i++){
        squared[i] = derivative[i] * derivative[i];
    }

    const unsigned int window_size = sampling_rate_hz * 0.15;
    std::vector<float> integrated(squared.size(), 0.0f);

    float sliding_sum{};
    for (int i = 0; i < window_size; i++) {
        sliding_sum += squared[i];
    }
    integrated[window_size - 1] = sliding_sum / window_size;

    for (size_t i = window_size; i < squared.size(); i++){
        sliding_sum = sliding_sum - squared[i-window_size] + squared[i];
        integrated[i] = sliding_sum / window_size;
    }   
}