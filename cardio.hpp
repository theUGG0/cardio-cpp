#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

struct DataSample
{
    uint32_t timestamp_ms;
    float reading_mv;
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
        std::array<T, BS>::iterator begin();
        std::array<T, BS>::iterator end();
        T& operator[] (size_t n);
};

template<typename T, size_t BS>
T& circular_buffer<T, BS>::operator[] (size_t n){
    assert(n < BS); // just incase
    if (n >= BS) n = BS-1; // clamp out of bounds.. feels not that pretty 
    size_t new_idx =  (BS + m_idx - 1 - n)%BS;
    return m_data[new_idx];
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

template<typename T, size_t BS>
std::array<T, BS>::iterator circular_buffer<T, BS>::begin(){
    return m_data.begin();
}

template<typename T, size_t BS>
std::array<T, BS>::iterator circular_buffer<T, BS>::end(){
    return m_data.end();
}

class HeartRateAnalyzer
{
    private:
        unsigned int m_sampling_rate_hz{200};

    public:
        HeartRateAnalyzer() = default;
        HeartRateAnalyzer(int sample_rate_hz) : m_sampling_rate_hz(sample_rate_hz) {};
        std::vector<DataSample> _pam_tompkins(std::vector<float>& raw_data_mv);

        int add_data(std::vector<float> data);
        int push_data(float sample, uint32_t timestamp_ms);

        HRVStatus get_status(uint32_t window_start_ms, uint32_t window_end_ms);
        std::vector<DataSample> get_beats(uint32_t window_start_ms, uint32_t window_end_ms);
};

// based on https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm
// and https://doi.org/10.1109%2FTBME.1985.325532
template<size_t SAMPLERATE> // recommended window size samplerate * 0.15, but can be empirical
class PanTompkins
{
    private:

        static constexpr int m_integrate_window{static_cast<size_t>(SAMPLERATE * 0.35)};

        // each buffer gets its name from what stage its buffering FOR
        circular_buffer<DataSample, 13> m_lowpass_buff{};
        circular_buffer<DataSample, 33> m_highpass_buff{};
        circular_buffer<DataSample, static_cast<size_t>(m_integrate_window*2)> m_sqr_derivative_buff{};
        circular_buffer<DataSample, static_cast<size_t>(m_integrate_window)> m_integrate_buff{};
        double m_sliding_sum{};
        circular_buffer<DataSample, 4> m_detector_buff{};

        void m_lowpass();
        void m_highpass();
        void m_sqr_derivative();
        void m_integrate();

        static constexpr int m_learning_needed_samples{SAMPLERATE*2};
        static const uint32_t m_refractory_period{200};

        uint32_t m_last_QRS_timestamp_ms{0};
        // running estimates for integrated signal
        int m_integrated_init_buff_len{};
        std::array<float, m_learning_needed_samples> m_thresh_i_learning_buff{}; // samplerate * 2 for 2 secs of samples
        double m_SPKI{}; // signal peak
        double m_NPKI{}; // noise peak
        double m_thresholdI{std::numeric_limits<double>::quiet_NaN()};
        
        // running estimates for highpass filtered signal  
        int m_filtered_init_buff_len{};
        std::array<float, m_learning_needed_samples> m_thresh_f_learning_buff{};
        double m_SPKF{};
        double m_NPKF{};
        double m_thresholdF{std::numeric_limits<double>::quiet_NaN()};

        void m_initialise_thresholds(double &SPK, double &NPK, double &threshold, std::array<float, m_learning_needed_samples> buffer);
        void m_update_thresholds();

        std::optional<DataSample> m_peak_detector(); //TODO: paper says total delay should be 24 samples i think. need to account for
    public:
        PanTompkins() = default;
        ~PanTompkins() = default;
        std::optional<DataSample> push_data(DataSample new_data);
};

template<size_t SAMPLERATE>
std::optional<DataSample> PanTompkins<SAMPLERATE>::push_data(DataSample new_data){
    m_lowpass_buff.push_value(new_data);

    if(m_lowpass_buff.full()) m_lowpass();
    if(m_highpass_buff.full()) m_highpass();

    if(std::isnan(m_thresholdF) && m_filtered_init_buff_len >= m_learning_needed_samples){
        m_initialise_thresholds(m_SPKF, m_NPKF, m_thresholdF, m_thresh_f_learning_buff);
    }

    if(m_sqr_derivative_buff.full()) m_sqr_derivative();
    if(m_integrate_buff.full()) m_integrate();

    if(std::isnan(m_thresholdI) && m_integrated_init_buff_len >= m_learning_needed_samples){
        m_initialise_thresholds(m_SPKI, m_NPKI, m_thresholdI, m_thresh_i_learning_buff);
    }

    if(m_detector_buff.full()) return m_peak_detector();

    return std::nullopt;
}

template<size_t SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_lowpass(){
        // filter delay is 2
        m_highpass_buff.push_value(DataSample(m_lowpass_buff[2].timestamp_ms,
            m_lowpass_buff[0].reading_mv 
            - 2*m_lowpass_buff[6].reading_mv 
            + m_lowpass_buff[12].reading_mv 
            + 2*m_highpass_buff[1].reading_mv
            - m_highpass_buff[2].reading_mv));
        m_highpass_buff[0].reading_mv /= 32; // normalize gain
}

template<size_t SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_highpass(){

    float highpass_val = m_highpass_buff[1].reading_mv
        - m_highpass_buff[0].reading_mv/32.0f // div by 32 comes from the errata corrige of the paper i think
        + m_highpass_buff[16].reading_mv
        - m_highpass_buff[17].reading_mv
        + m_highpass_buff[32].reading_mv/32.0f;

    if(m_filtered_init_buff_len < m_learning_needed_samples){
        m_thresh_f_learning_buff[m_filtered_init_buff_len++] = highpass_val;
    }
    // filter delay is 16
    m_sqr_derivative_buff.push_value(DataSample(m_highpass_buff[16].timestamp_ms, highpass_val));
}

template<size_t SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_sqr_derivative(){
    m_integrate_buff.push_value(DataSample(m_sqr_derivative_buff[2].timestamp_ms,
        pow(
            0.125 * (2*m_sqr_derivative_buff[0].reading_mv
                    + m_sqr_derivative_buff[1].reading_mv 
                    - m_sqr_derivative_buff[3].reading_mv 
                    - 2*m_sqr_derivative_buff[4].reading_mv),
             2)));
    
    m_sliding_sum += m_sqr_derivative_buff[0].reading_mv;
}

template<size_t SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_integrate(){
    
    float integrated_val = m_sliding_sum / m_integrate_window;

    if(m_integrated_init_buff_len < m_learning_needed_samples){
        m_thresh_i_learning_buff[m_integrated_init_buff_len++] = integrated_val;
    }

    // the timestamp here is iffy.. need to think through the filter delay
    m_detector_buff.push_value(DataSample(m_sqr_derivative_buff[m_integrate_window/2].timestamp_ms, integrated_val));
    m_sliding_sum -= m_sqr_derivative_buff[m_integrate_window-1].reading_mv;
}

template<size_t SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_initialise_thresholds(double &SPK, double &NPK, double &threshold, std::array<float, m_learning_needed_samples> buffer){
    SPK = *std::max_element(buffer.begin(), buffer.end());
    NPK = std::accumulate(buffer.begin(), buffer.end(), 0.0l) / m_learning_needed_samples;
    threshold = NPK+0.25*(SPK-NPK);

    std::cout << "THRESHOLDS: " << SPK << " " << NPK << " " << threshold << "\n";
}

template<size_t SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_update_thresholds(){
    m_thresholdI = m_NPKI + 0.25 * (m_SPKI - m_NPKI);
    m_thresholdF = m_NPKF + 0.25 * (m_SPKF - m_NPKF);
}


// TODO: Implement searchback with half of thresholdI1 at 1.66x RR interval passed with no peaks.
// TODO: paper says total delay should be 24 samples i think. need to account for
template<size_t SAMPLERATE>
std::optional<DataSample> PanTompkins<SAMPLERATE>::m_peak_detector(){

    auto get_new_PK_thresh = [](double peak, double thresh){return 0.125 * peak + 0.875 * thresh;};
    std::optional<DataSample> ret_QRS{};

    if (m_detector_buff[2].reading_mv < m_detector_buff[1].reading_mv && m_detector_buff[1].reading_mv > m_detector_buff[0].reading_mv) {
        DataSample peakI = m_detector_buff[1];

        // account for refractory period
        if (m_last_QRS_timestamp_ms > 0 && (peakI.timestamp_ms - m_last_QRS_timestamp_ms <= m_refractory_period)){
            m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
            m_update_thresholds();
            return ret_QRS;
        }

        // if peak, check for it in bandpass filtered signal
        if (peakI.reading_mv > m_thresholdI){

            DataSample peakF; // TODO: add some check here for if the peak isnt found
            bool foundPeakF{false};
            for (size_t i{0}; i < m_integrate_window*2; i++){
                if (m_sqr_derivative_buff[i].timestamp_ms == peakI.timestamp_ms){
                    peakF = m_sqr_derivative_buff[i];
                    foundPeakF = true;
                    break;
                }
            }
            if(!foundPeakF){
                m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
                m_update_thresholds();
                return ret_QRS;
            }
            
            if (std::abs(peakF.reading_mv) > m_thresholdF){
                ret_QRS = peakF;

                m_last_QRS_timestamp_ms = ret_QRS->timestamp_ms;

                m_SPKF = get_new_PK_thresh(peakF.reading_mv, m_SPKF);
                m_SPKI = get_new_PK_thresh(peakI.reading_mv, m_SPKI);
            }

            // else wasnt an R peak, update noise
            else {
                m_NPKF = get_new_PK_thresh(peakF.reading_mv, m_NPKF);
                m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
            }
        }
        else {
            m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
        }
    }

    m_update_thresholds();

    return ret_QRS;
}  


std::vector<DataSample> HeartRateAnalyzer::_pam_tompkins(std::vector<float>& raw_data_mv){
    std::vector<float> lowpass(raw_data_mv.size(), 0.0f);

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
/*
    std::unordered_set<size_t> r_peak_set(R_peaks.begin(), R_peaks.end());

    for (size_t i = 0; i < integrated.size(); i++){
        file << i << "," << integrated[i] << ",";
        
        if (r_peak_set.count(i) > 0){
            file << "N"; 
        }
        
        file << "\n";
    }

    file.close();
*/
    return {};
}