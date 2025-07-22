class Cache:
    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, 0

class DynamicCache(Cache):
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__()
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()
            ):
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (
            len(self.key_cache) == 0
            or len(self.key_cache) <= layer_idx
            or not self.key_cache[layer_idx].numel()
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[tuple[tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None

_is_offline_mode = huggingface_hub.constants.HF_HUB_OFFLINE

def is_offline_mode():
    return _is_offline_mode

def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    **kwargs,
) -> Optional[str]:
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    file = file[0] if file is not None else file
    return file

def cached_files(
    path_or_repo_id: Union[str, os.PathLike],
    filenames: list[str],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    print(user_agent)
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        token = use_auth_token

    if is_offline_mode() and not local_files_only:
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    full_filenames = [os.path.join(subfolder, file) for file in filenames]

    path_or_repo_id = str(path_or_repo_id)
    existing_files = []
    for filename in full_filenames:
        if os.path.isdir(path_or_repo_id):
            resolved_file = os.path.join(path_or_repo_id, filename)
            if not os.path.isfile(resolved_file):
                if _raise_exceptions_for_missing_entries and filename != os.path.join(subfolder, "config.json"):
                    revision_ = "main" if revision is None else revision
                else:
                    return None
            existing_files.append(resolved_file)

    if len(existing_files) == len(full_filenames):
        return existing_files

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    existing_files = []
    file_counter = 0
    if _commit_hash is not None and not force_download:
        for filename in full_filenames:
            resolved_file = try_to_load_from_cache(
                path_or_repo_id, filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
            )
            if resolved_file is not None:
                if resolved_file is not _CACHED_NO_EXIST:
                    file_counter += 1
                    existing_files.append(resolved_file)
                elif not _raise_exceptions_for_missing_entries:
                    file_counter += 1
    if file_counter == len(full_filenames):
        return existing_files if len(existing_files) > 0 else None

    #user_agent = http_user_agent(user_agent)
    user_agent = None
    if len(full_filenames) == 1:
        hf_hub_download(
            path_or_repo_id,
            filenames[0],
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    else:
        snapshot_download(
            path_or_repo_id,
            allow_patterns=full_filenames,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )

    resolved_files = [
        _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    ]
    if any(file is None for file in resolved_files) and _raise_exceptions_for_missing_entries:
        missing_entries = [original for original, resolved in zip(full_filenames, resolved_files) if resolved is None]
        if len(resolved_files) == 1 and missing_entries[0] == os.path.join(subfolder, "config.json"):
            return None
        revision_ = "main" if revision is None else revision

    resolved_files = [file for file in resolved_files if file is not None]
    resolved_files = None if len(resolved_files) == 0 else resolved_files

    return resolved_files