#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <unordered_set>
#include <vector>

struct HeartBeat
{
    uint32_t timestamp_ms;
    uint32_t peak_mv;
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

template<typename T, size_t BS>
class circular_buffer
{
    std::array<T, BS> m_data{};
    size_t m_idx{};
    size_t m_fill_count{};

    public:
        circular_buffer() = default;
        circular_buffer(std::array<T, BS> data, size_t len) : m_data(data), m_idx(len-1), m_fill_count(m_idx){};
        ~circular_buffer() = default; 
        void push_value(T n);
        size_t size();
        bool full();
        T operator[] (size_t n);
};

template<typename T, size_t BS>
T circular_buffer<T, BS>::operator[] (size_t n){
    if (n > BS) n = BS; // just incase
    size_t new_idx =  m_idx - 1 - n;
    return m_data[(BS+new_idx)%BS];
}

template<typename T, size_t BS>
void circular_buffer<T, BS>::push_value(T n){
    if (m_fill_count < BS) m_fill_count++;

    m_data[m_idx] = n;
    m_idx = (m_idx + 1) % BS;
}

template<typename T, size_t BS>
size_t circular_buffer<T, BS>::size(){
    return m_fill_count;
}

template<typename T, size_t BS>
bool circular_buffer<T, BS>::full(){
    return m_fill_count >= BS;
}

class HeartRateAnalyzer
{
    private:
        unsigned int m_sampling_rate_hz{200};

    public:
        HeartRateAnalyzer() = default;
        HeartRateAnalyzer(int sample_rate_hz) : m_sampling_rate_hz(sample_rate_hz) {};
        std::vector<HeartBeat> _pam_tompkins(std::vector<float>& raw_data_mv);

        int add_data(std::vector<float> data);
        int push_data(float sample, uint32_t timestamp_ms);

        HRVStatus get_status(uint32_t window_start_ms, uint32_t window_end_ms);
        std::vector<HeartBeat> get_beats(uint32_t window_start_ms, uint32_t window_end_ms);
};

// based on https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm
class PanTompkins
{
    private:

        // use queues as circular buffers between processing steps for memory optimisation
        std::queue<float> m_temp_raw_buff{}; // needed if first pushed data count < lowpass needed data count 
        std::queue<float> m_lowpass_buff{};
        std::queue<float> m_highpass_buff{};
        std::queue<float> m_sqr_derivative_buff{};
        std::queue<float> m_integrate_buff{};

        void m_lowpass();
        void m_highpass();
        void m_sqr_derivative();
        void m_integrate();

        // running estimates for integrated signal 
        double m_max_integrated{};
        double m_SPKI = 0.0;
        double m_NPKI = 0.0;
        double thresholdI1{}; //0.35 * m_max_integrated;
        
        // running estimates for filtered signal  
        double max_filtered{};
        double SPKF = 0.0;
        double NPKF = 0.0;
        double thresholdF1{}; // 0.35 * max_filtered;

        std::optional<HeartBeat> m_peak_detector();

    public:
        std::optional<float> push_data(std::vector<float>& raw_data_mv);
};

/*
std::optional<float> PanTompkins::push_data(std::vector<float>& raw_data_mv){
    // check if enough data is in filter pipeline stage buffers for processing to take place
    if (m_temp_raw_buff.size() >= 12) {
        m_lowpass();
    }
}*/


std::vector<HeartBeat> HeartRateAnalyzer::_pam_tompkins(std::vector<float>& raw_data_mv){
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

    const unsigned int window_size = m_sampling_rate_hz * 0.15;
    std::vector<float> integrated(squared.size(), 0.0f);

    float sliding_sum{};
    for (unsigned int i = 0; i < window_size; i++) {
        sliding_sum += squared[i];
    }
    integrated[window_size - 1] = sliding_sum / window_size;

    for (size_t i = window_size; i < squared.size(); i++){
        sliding_sum = sliding_sum - squared[i-window_size] + squared[i];
        integrated[i] = sliding_sum / window_size;
    }

    std::vector<size_t> R_peaks{};
    
    // running estimates for integrated signal
    double max_integrated = *std::max_element(integrated.begin(), integrated.end());
    double SPKI = 0.0;
    double NPKI = 0.0;
    double thresholdI1 = 0.35 * max_integrated;
    
    // running estimates for filtered signal  
    double max_filtered = std::max(*std::max_element(highpass.begin(), highpass.end()),
                                   std::abs(*std::min_element(highpass.begin(), highpass.end())));
    double SPKF = 0.0;
    double NPKF = 0.0;
    double thresholdF1 = 0.35 * max_filtered;
    
    const size_t refractory_period = m_sampling_rate_hz * 0.2;
    size_t last_qrs_index = 0;
    bool first_peak = true;
    

    // TODO: Implement searchback with half of thresholdI1 at 1.66x RR interval passed with no peaks.
    // TODO: Fix offset issues with detected peaks (seems to be around 30 samples)

    for (size_t i = 2; i < integrated.size(); i++){
    // look for peak in integrated signal
        if (integrated[i-2] < integrated[i-1] && integrated[i-1] > integrated[i]){
            
            float peakI = integrated[i-1];
            size_t peak_idx_integrated = i - 1;
            
            // skip peaks during refractory period
            if (last_qrs_index > 0 && (peak_idx_integrated - last_qrs_index <= refractory_period)){
                continue;
            }
            
            // Now process the peak
            if (peakI > thresholdI1){
                
                // Search for corresponding peak in filtered signal
                size_t search_start = (peak_idx_integrated > 50) ? peak_idx_integrated - 50 : 0;
                size_t search_end = std::min(peak_idx_integrated + 50, highpass.size() - 1);
                
                float max_abs_filtered = 0.0f;
                size_t peak_idx_filtered = peak_idx_integrated;
                
                for (size_t j = search_start; j <= search_end; j++){
                    if (std::abs(highpass[j]) > max_abs_filtered){
                        max_abs_filtered = std::abs(highpass[j]);
                        peak_idx_filtered = j;
                    }
                }
                
                float peakF = std::abs(highpass[peak_idx_filtered]);
                
                // check if found filtered signal is over filtered thresh
                // if yes, its a QRS 
                if (peakF > thresholdF1){

                    R_peaks.push_back(peak_idx_filtered);
                    last_qrs_index = peak_idx_integrated;
                    
                    if (first_peak){
                        SPKI = peakI;
                        SPKF = peakF;
                        first_peak = false;
                    } else {
                        SPKI = 0.125 * peakI + 0.875 * SPKI;
                        SPKF = 0.125 * peakF + 0.875 * SPKF;
                    }
                } else {
                    // if wasnt over filtered thresh it had to be noise
                    NPKI = 0.125 * peakI + 0.875 * NPKI;
                    NPKF = 0.125 * peakF + 0.875 * NPKF;
                }
                
            } else {
                // if peak wasnt over interated thresh it had to be noise
                NPKI = 0.125 * peakI + 0.875 * NPKI;
            }
                
            // update integrated and filtered signal threshold if running estimates are over 0
            if (SPKI > 0 && NPKI > 0){
                thresholdI1 = NPKI + 0.25 * (SPKI - NPKI);
            }
            
            if (SPKF > 0 && NPKF > 0){
                thresholdF1 = NPKF + 0.25 * (SPKF - NPKF);
            }
        }
    }
    
    std::cout << "Detected " << R_peaks.size() << " QRS complexes" << std::endl;

    std::ofstream file{"testout.csv"};
    file << "index,integrated,r_peak_marker\n";

    std::unordered_set<size_t> r_peak_set(R_peaks.begin(), R_peaks.end());

    for (size_t i = 0; i < integrated.size(); i++){
        file << i << "," << integrated[i] << ",";
        
        if (r_peak_set.count(i) > 0){
            file << "N"; 
        }
        
        file << "\n";
    }

    file.close();

    return {};
}