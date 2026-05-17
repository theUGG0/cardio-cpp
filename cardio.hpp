#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

template<unsigned int SAMPLERATE>
class PanTompkins;

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

template<unsigned int SAMPLERATE> 
class HeartRateAnalyzer
{
    private:
        std::vector<DataSample> m_rpeaks{};
        PanTompkins<SAMPLERATE> m_pt{};
        uint32_t m_avg_RR_interval{};

        void m_update_avg_RR();

    public:
        HeartRateAnalyzer() : m_pt{PanTompkins<SAMPLERATE>()} {};

        int add_data(std::vector<float> data);
        int push_data(float sample, uint32_t timestamp_ms);

        HRVStatus get_status(uint32_t window_start_ms, uint32_t window_end_ms);
        std::vector<DataSample> get_beats(uint32_t window_start_ms, uint32_t window_end_ms) const;
};

template<unsigned int SAMPLERATE> 
void HeartRateAnalyzer<SAMPLERATE>::m_update_avg_RR(){
    if (m_rpeaks.size() < 2) return;  // need at least 2 peaks for 1 interval

    size_t peaks_to_use = std::min<size_t>(m_rpeaks.size(), 8);

    std::vector<uint32_t> intervals;
    intervals.reserve(peaks_to_use - 1);
    for (size_t i = 1; i < peaks_to_use; ++i){
        auto newer = std::prev(m_rpeaks.end(), i);
        auto older = std::prev(m_rpeaks.end(), i + 1);
        intervals.push_back(newer->timestamp_ms - older->timestamp_ms);
    }

    uint64_t sum = std::accumulate(intervals.begin(), intervals.end(), uint64_t{0});
    m_avg_RR_interval = static_cast<uint32_t>(sum / intervals.size());
}

template<unsigned int SAMPLERATE> 
int HeartRateAnalyzer<SAMPLERATE>::push_data(float sample, uint32_t timestamp_ms){
    std::optional<DataSample> n_dat = m_pt.push_data(DataSample(timestamp_ms, sample));
    if(n_dat.has_value()){
        m_rpeaks.push_back(n_dat.value());
        m_update_avg_RR();
        m_pt.setAvgRR(m_avg_RR_interval);
        return 1;
    }
    return 0;
}

template<unsigned int SAMPLERATE>
std::vector<DataSample> HeartRateAnalyzer<SAMPLERATE>::get_beats(uint32_t window_start_ms, uint32_t window_end_ms) const {

    auto cmp = [](const DataSample& s, uint32_t t){ return s.timestamp_ms < t; };

    auto first = std::lower_bound(m_rpeaks.begin(), m_rpeaks.end(), window_start_ms, cmp);
    auto last = std::lower_bound(m_rpeaks.begin(), m_rpeaks.end(), window_end_ms, cmp);

    return std::vector<DataSample>(first, last);
}



// based on https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm
// and https://doi.org/10.1109%2FTBME.1985.325532
template<unsigned int SAMPLERATE>
class PanTompkins
{
    private:

        static constexpr int m_integrate_window{static_cast<size_t>(SAMPLERATE * 0.15)};

        // each buffer gets its name from what stage its buffering FOR
        circular_buffer<DataSample, 13> m_lowpass_buff{};
        circular_buffer<DataSample, 33> m_highpass_buff{};
        circular_buffer<DataSample, static_cast<size_t>(m_integrate_window*2)> m_sqr_derivative_buff{};
        circular_buffer<DataSample, static_cast<size_t>(m_integrate_window)> m_integrate_buff{};
        double m_sliding_sum{};
        circular_buffer<DataSample, 4> m_detector_buff{};

        uint32_t m_avg_RR_interval{};

        DataSample m_max_since_R{DataSample(0,0)};
        bool m_backsearch_done{false};
        circular_buffer<DataSample, 4> m_backsearch_peak_buff{};

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

        std::optional<DataSample> m_peak_threshold(const DataSample& peakI, double thresholdI, double thresholdF, bool is_searchback);        std::optional<DataSample> m_peak_detector();
    
    public:
        PanTompkins() = default;
        ~PanTompkins() = default;
        std::optional<DataSample> push_data(DataSample new_data);
        void setAvgRR(uint32_t interv);

};

template<unsigned int SAMPLERATE>
void PanTompkins<SAMPLERATE>::setAvgRR(uint32_t interv){
    m_avg_RR_interval = interv;
}

template<unsigned int SAMPLERATE>
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

template<unsigned int SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_lowpass(){
        // filter delay is 2
        m_highpass_buff.push_value(DataSample(m_lowpass_buff[2].timestamp_ms,
            m_lowpass_buff[0].reading_mv 
            - 2*m_lowpass_buff[6].reading_mv 
            + m_lowpass_buff[12].reading_mv 
            + 2*m_highpass_buff[0].reading_mv
            - m_highpass_buff[1].reading_mv));
        m_highpass_buff[0].reading_mv /= 32; // normalize gain
}

template<unsigned int SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_highpass(){

    float highpass_val = m_highpass_buff[16].reading_mv
        - m_highpass_buff[0].reading_mv / 32.0f
        + m_highpass_buff[32].reading_mv / 32.0f
        - m_sqr_derivative_buff[0].reading_mv;

    if(m_filtered_init_buff_len < m_learning_needed_samples){
        m_thresh_f_learning_buff[m_filtered_init_buff_len++] = highpass_val;
    }
    // filter delay is 16
    m_sqr_derivative_buff.push_value(DataSample(m_highpass_buff[16].timestamp_ms, highpass_val));
}

template<unsigned int SAMPLERATE>
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

template<unsigned int SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_integrate(){
    
    float integrated_val = m_sliding_sum / m_integrate_window;

    if(m_integrated_init_buff_len < m_learning_needed_samples){
        m_thresh_i_learning_buff[m_integrated_init_buff_len++] = integrated_val;
    }

    // the timestamp here is iffy.. need to think through the filter delay
    m_detector_buff.push_value(DataSample(m_sqr_derivative_buff[m_integrate_window/2].timestamp_ms, integrated_val));
    m_sliding_sum -= m_sqr_derivative_buff[m_integrate_window-1].reading_mv;
}

template<unsigned int SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_initialise_thresholds(double &SPK, double &NPK, double &threshold, std::array<float, m_learning_needed_samples> buffer){
    SPK = *std::max_element(buffer.begin(), buffer.end());
    NPK = std::accumulate(buffer.begin(), buffer.end(), 0.0l) / m_learning_needed_samples;
    threshold = NPK+0.25*(SPK-NPK);

    // DEBUG
    std::cout << "THRESHOLDS: " << SPK << " " << NPK << " " << threshold << "\n";
}

template<unsigned int SAMPLERATE>
void PanTompkins<SAMPLERATE>::m_update_thresholds(){
    m_thresholdI = m_NPKI + 0.25 * (m_SPKI - m_NPKI);
    m_thresholdF = m_NPKF + 0.25 * (m_SPKF - m_NPKF);
}

template<unsigned int SAMPLERATE>
std::optional<DataSample> PanTompkins<SAMPLERATE>::m_peak_threshold(const DataSample& peakI, double thresholdI, double thresholdF, bool is_searchback){

    auto get_new_PK_thresh = [](double peak, double thresh){return 0.125 * peak + 0.875 * thresh;};
    std::optional<DataSample> ret_QRS{};

    // Refractory period — same as before, but only update noise on the normal path.
    if (m_last_QRS_timestamp_ms > 0 && (peakI.timestamp_ms - m_last_QRS_timestamp_ms <= m_refractory_period)){
        if (!is_searchback) {
            m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
            m_update_thresholds();
        }
        return ret_QRS;
    }

    if (peakI.reading_mv > thresholdI){

        DataSample peakF;
        bool foundPeakF{false};
        for (size_t i{0}; i < m_integrate_window*2; i++){
            if (m_sqr_derivative_buff[i].timestamp_ms == peakI.timestamp_ms){
                peakF = m_sqr_derivative_buff[i];
                foundPeakF = true;
                break;
            }
        }
        if(!foundPeakF){
            if (!is_searchback) {
                m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
                m_update_thresholds();
            }
            return ret_QRS;
        }

        if (std::abs(peakF.reading_mv) > thresholdF){
            ret_QRS = peakF;

            m_last_QRS_timestamp_ms = ret_QRS->timestamp_ms;
            m_max_since_R = DataSample{0, 0.0f};

            // Confirmed beat — always update signal peak estimates.
            m_SPKF = get_new_PK_thresh(peakF.reading_mv, m_SPKF);
            m_SPKI = get_new_PK_thresh(peakI.reading_mv, m_SPKI);
            m_update_thresholds();
        }
        else if (!is_searchback) {
            // Crossed I but not F, and we're on the normal path — noise.
            m_NPKF = get_new_PK_thresh(peakF.reading_mv, m_NPKF);
            m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
            m_update_thresholds();
        }
    }
    else if (!is_searchback) {
        m_NPKI = get_new_PK_thresh(peakI.reading_mv, m_NPKI);
        m_update_thresholds();
    }

    return ret_QRS;
}

// TODO: Implement searchback with half of thresholdI1 at 1.66x RR interval passed with no peaks.
template<unsigned int SAMPLERATE>
std::optional<DataSample> PanTompkins<SAMPLERATE>::m_peak_detector(){

    std::optional<DataSample> ret_QRS{};

    // Always track running max since last confirmed R.
    if (m_max_since_R.reading_mv < m_detector_buff[0].reading_mv) {
        m_max_since_R = m_detector_buff[0];
    }

    if (m_detector_buff[2].reading_mv < m_detector_buff[1].reading_mv
        && m_detector_buff[1].reading_mv > m_detector_buff[0].reading_mv) {
        ret_QRS = m_peak_threshold(m_detector_buff[1], m_thresholdI, m_thresholdF, false);
    }

    if (!ret_QRS
        && m_avg_RR_interval > 0
        && (m_detector_buff[0].timestamp_ms - m_last_QRS_timestamp_ms) > 1.66 * m_avg_RR_interval)
    {
        ret_QRS = m_peak_threshold(m_max_since_R, m_thresholdI * 0.5, m_thresholdF * 0.5, true);
    }

    return ret_QRS;
}