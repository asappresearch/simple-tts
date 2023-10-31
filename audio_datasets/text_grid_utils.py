from praatio import textgrid
import os

data_path = '/persist/data/aligned_librispeech/test-clean/1089/1089-134686-0000.TextGrid'

tg = textgrid.openTextgrid(data_path, False)
def get_word_intervals(textgrid_path):
    tg = textgrid.openTextgrid(textgrid_path, False)
    return tg.getTier("words").entries

def get_partial_transcript(textgrid_path, transcript_end_time=3):
    intervals = get_word_intervals(textgrid_path)
    word_list = []
    end_time = 0
    for interval in intervals:
        if interval.end > transcript_end_time:
            break
        word_list.append(interval.label)
        end_time = interval.end
    return {'transcript': ' '.join(word_list), 
            'end_time': end_time}


