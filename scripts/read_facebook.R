library(tidyverse)
library(jsonlite)
library(anytime)   # to convert epoch time to date object
library(janitor)   # clean column names
library(cld2)      # detect language
library(lubridate) # dealing with time and date stuff
library(progress)  # progress bars
library(here)

read_all_messages <- function(data_path) {
  dirs <- list.files(here::here(data_path, "messages/inbox"))
  message_dirs <- dirs[str_starts(dirs, "[\\p{Ll}]")]
  chat_names <- str_extract(message_dirs, "^[^_]+(?=_)")
  chat_names[is.na(chat_names)] <- message_dirs[is.na(chat_names)]
  pb <- progress_bar$new(format = "[:bar] :percent eta: :eta",
                         clear = FALSE,
                         total = length(message_dirs))
  pb$tick(0)
  res <- message_dirs %>%
    setNames(message_dirs) %>%
    map_dfr(.f = function(x) read_chat_with_progress(x, data_path = data_path, pb = pb),
            .id = "chatname")
  res
}

read_chat_with_progress <- function(chat, data_path, pb){
  pb$tick()
  read_chat(chat, data_path = data_path)
}

read_chatfile <- function(path) {
  jsonlite::fromJSON(path, flatten = TRUE)
}

clean_messages <- function(json_frame) {
  participants <- paste(json_frame$participants$name, collapse = "_")
  num_participants <- max(json_frame$messages %>% distinct(sender_name) %>% nrow, 
                          length(json_frame$participants$name ))
  nr <- nrow(json_frame$messages)
  cols <- c(audio_files = list(rep(list(NULL), nr) ), 
            files = list(rep(list(NULL), nr) ), 
            photos = list(rep(list(NULL), nr) ),
            gifs = list(rep(list(NULL), nr) ),
            reactions = list(rep(list(NULL), nr) ),
            participants = participants,
            sticker.uri = NA_character_,
            share.link = NA_character_,
            call_duration = NA_integer_,
            num_participants = num_participants,
            content = NA_character_)
  
  messages <- json_frame$messages %>%
    as_tibble %>%
    add_column( !!!cols[setdiff(names(cols), names(.))]) %>%
    filter(type != "Subscribe" & type != "Unsubscribe") %>%
    mutate(num_photos = map_dbl(photos, .f = length ),
           num_audio = map_dbl(audio_files, .f = length ),
           num_reactions = map_dbl(reactions, .f = length ),
           num_files = map_dbl(files, .f = length), 
           num_gifs = map_dbl(gifs, .f = length ),
           timestamp = anytime(timestamp_ms / 1000),
           is_sticker = !is.na(sticker.uri),
           content = if_else(str_detect(content, 
                                        "\\w sent an attachment.|\\w sent a location.|\\w sent an event link.|\\w sent a link."),
                             share.link, content),
           word_count = str_count(content, "\\S+"),
           chars = nchar(content),
           lang = cld2::detect_language(content)) %>%
    janitor::clean_names() %>%
    select(sender_name, type, content, call_duration, share_link, participants, 
           num_photos, num_audio, num_reactions, num_files, num_gifs,
           timestamp, is_sticker, word_count, chars, lang, num_participants) 
  messages
}

aggregate_messages <- function(messages) {
  mess_agg <- messages %>%
    group_by(participants,
             num_participants,
             day = lubridate::floor_date(timestamp, unit="days"), 
             lang, sender_name) %>%
    summarise(num_messages = n(),
              num_words = sum(word_count),
              num_chars = sum(chars),
              num_links = sum(!is.na(share_link)),
              num_photos = sum(num_photos),
              num_files = sum(num_files),
              num_sticker = sum(is_sticker),
              num_gifs = sum(num_gifs),
              num_audio = sum(num_audio),
              num_calls = sum(!is.na(call_duration)),
              num_reactions = sum(num_reactions),
              .groups = "drop")
  mess_agg
}

read_chatdata_agg <- function(path) {
  read_chatfile(path) %>%
    clean_messages() %>%
    aggregate_messages
}
read_chatdata <- function(path) {
  read_chatfile(path) %>%
    clean_messages() 
}

read_chat <- function(chat, data_path, agg=TRUE) {
  chat_files <- list.files(here::here(data_path, "messages/inbox", chat), 
                           pattern="*.json",
                           full.names = TRUE)
  
  if (agg) {df <- map_dfr(chat_files, .f = read_chatdata_agg)}
  else {df <- map_dfr(chat_files, .f = read_chatdata)}
  df
}